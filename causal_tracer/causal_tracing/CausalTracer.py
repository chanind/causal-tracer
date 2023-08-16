from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence
import torch
from torch import nn
from tokenizers import Tokenizer

from causal_tracer.causal_tracing.AsyncTracePatchProcessor import (
    AsyncTracePatchProcessor,
)
from causal_tracer.causal_tracing.PseudoFuture import PseudoFuture
from causal_tracer.causal_tracing.guess_subject import guess_subject
from causal_tracer.causal_tracing.pick_noise_level import pick_noise_level
from causal_tracer.lib.TraceLayerDict import TraceLayerDict
from causal_tracer.lib.layer_matching import (
    LayerMatcher,
    collect_matching_layers,
    get_layer_name,
)
from causal_tracer.lib.torch_utils import untuple_tensor
from causal_tracer.lib.constants import DEFAULT_DEVICE

from causal_tracer.lib.token_utils import (
    decode_tokens,
    find_token_range,
    make_inputs,
    predict_from_input,
)
from causal_tracer.lib.util import batchify
from .HiddenFlow import HiddenFlow, LayerKind


@dataclass
class BaseOutputs:
    answer_token_ids: torch.Tensor
    base_scores: torch.Tensor
    uncorrupted_layer_outputs: dict[int, OrderedDict[str, torch.Tensor]]
    attention_masks: list[torch.Tensor]


@dataclass
class HiddenFlowQuery:
    text: str
    subject: Optional[str] = None
    override_answer_id: Optional[int] = None


class CausalTracer:
    model: nn.Module
    tokenizer: Tokenizer
    embed_layername: str
    hidden_layers_matcher: LayerMatcher
    mlp_layers_matcher: LayerMatcher
    attn_layers_matcher: LayerMatcher
    noise: float
    device: torch.device
    noise_calculation_subjects: Optional[list[str]]

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        embed_layername: str = "transformer.wte",
        hidden_layers_matcher: LayerMatcher = "transformer.h.{num}",
        mlp_layers_matcher: LayerMatcher = "transformer.h.{num}.mlp",
        attn_layers_matcher: LayerMatcher = "transformer.h.{num}.attn",
        noise: Optional[float] = None,
        noise_calculation_subjects: Optional[list[str]] = None,
        device: torch.device = DEFAULT_DEVICE,
    ) -> None:
        model.eval()
        self.model = model
        self.tokenizer = tokenizer
        self.embed_layername = embed_layername
        self.hidden_layers_matcher = hidden_layers_matcher
        self.mlp_layers_matcher = mlp_layers_matcher
        self.attn_layers_matcher = attn_layers_matcher
        self.noise = noise or pick_noise_level(model, embed_layername)
        self.device = device
        self.noise_calculation_subjects = noise_calculation_subjects

    def count_layers(self) -> int:
        return len(collect_matching_layers(self.model, self.hidden_layers_matcher))

    def get_layer_name(self, layer_num: int, layer_kind: LayerKind) -> str:
        if layer_kind == "hidden":
            return get_layer_name(self.model, self.hidden_layers_matcher, layer_num)
        if layer_kind == "mlp":
            return get_layer_name(self.model, self.mlp_layers_matcher, layer_num)
        if layer_kind == "attn":
            return get_layer_name(self.model, self.attn_layers_matcher, layer_num)
        raise ValueError(f"Unknown layer kind {layer_kind}")

    def get_layer_names_of_kind(self, layer_kind: LayerKind) -> list[str]:
        return [
            self.get_layer_name(layer_num, layer_kind)
            for layer_num in range(self.count_layers())
        ]

    def trace_important_window(
        self,
        prompt: str,
        uncorrupted_layer_outputs: dict[str, torch.Tensor],
        subject_range: tuple[int, int],
        answer_token_id: torch.Tensor,
        kind: LayerKind,
        trace_patch_processor: AsyncTracePatchProcessor,
        ntoks: int,
        window: int = 10,
        token_range: Optional[tuple[int, int]] = None,
        start_layer: int = 0,
        end_layer: Optional[int] = None,
    ) -> list[list[PseudoFuture[torch.Tensor]]]:
        """
        This appears to restore multiple layers at once, in a window around the layer of interest.
        This seems to only be used for the hidden layer kind, not the mlp or attn layer kinds.

        In the ROME colab, it says the following:

        'Because MLP and attention make small residual contributions, to observe a causal effect in those cases,
        we need to restore several layers of contributions at once, which is done by trace_important_window'
        """
        num_layers = self.count_layers()
        table: list[list[PseudoFuture[torch.Tensor]]] = []
        for tnum in range(*(token_range or (0, ntoks))):
            row = []
            for layer in range(start_layer, end_layer or num_layers):
                layerlist = [
                    (tnum, self.get_layer_name(layer_num, kind))
                    for layer_num in range(
                        max(0, layer - window // 2),
                        min(num_layers, layer - (-window // 2)),
                    )
                ]
                res_future = trace_patch_processor.trace_with_patch(
                    prompt,
                    states_to_patch=layerlist,
                    uncorrupted_layer_outputs=uncorrupted_layer_outputs,
                    answer_token_id=answer_token_id,
                    subject_range=subject_range,
                )
                row.append(res_future)
            table.append(row)
        return table

    def trace_important_states(
        self,
        prompt: str,
        uncorrupted_layer_outputs: dict[str, torch.Tensor],
        subject_range: tuple[int, int],
        answer_token_id: torch.Tensor,
        ntoks: int,
        trace_patch_processor: AsyncTracePatchProcessor,
        start_layer: int = 0,
        end_layer: Optional[int] = None,
        token_range: Optional[tuple[int, int]] = None,
    ) -> list[list[PseudoFuture[torch.Tensor]]]:
        num_layers = self.count_layers()
        table: list[list[PseudoFuture[torch.Tensor]]] = []
        for tnum in range(*(token_range or (0, ntoks))):
            row = []
            for layer in range(start_layer, end_layer or num_layers):
                res_future = trace_patch_processor.trace_with_patch(
                    prompt,
                    states_to_patch=[(tnum, self.get_layer_name(layer, "hidden"))],
                    uncorrupted_layer_outputs=uncorrupted_layer_outputs,
                    answer_token_id=answer_token_id,
                    subject_range=subject_range,
                )
                row.append(res_future)
            table.append(row)
        return table

    def calculate_hidden_flow(
        self,
        prompt: str,
        subject: Optional[str] = None,
        answer_id_override: Optional[int] = None,
        samples: int = 10,
        window: int = 10,
        kind: LayerKind = "hidden",
        patch_seed: int = 1,
        batch_size: int = 32,
        start_layer: int = 0,
        end_layer: Optional[int] = None,
        patch_subject_tokens_only: bool = False,
    ) -> HiddenFlow:
        """
        Runs causal tracing over every token/layer combination in the network
        and returns a dictionary numerically summarizing the results.
        """
        return self.calculate_hidden_flows(
            [HiddenFlowQuery(prompt, subject, answer_id_override)],
            samples=samples,
            window=window,
            kind=kind,
            patch_seed=patch_seed,
            batch_size=batch_size,
            start_layer=start_layer,
            end_layer=end_layer,
            patch_subject_tokens_only=patch_subject_tokens_only,
        )[0]

    def calculate_base_outputs_batched(
        self,
        prompts: Sequence[str],
        kind: LayerKind,
        batch_size: int,
        show_progress: bool = False,
        move_to_cpu: bool = False,
        answer_id_overrides: list[tuple[int, int]] = [],
    ) -> BaseOutputs:
        # things will break if we try to process an empty batch, so just return empty results directly
        if len(prompts) == 0:
            return BaseOutputs(
                answer_token_ids=torch.empty(0),
                base_scores=torch.empty(0),
                uncorrupted_layer_outputs={},
                attention_masks=[],
            )
        answer_ids_outputs: list[torch.Tensor] = []
        base_scores_outputs: list[torch.Tensor] = []
        attention_masks: list[torch.Tensor] = []
        uncorrupted_layer_outputs: dict[
            int, OrderedDict[str, torch.Tensor]
        ] = defaultdict(OrderedDict)
        total_processed = 0
        for batch in batchify(prompts, batch_size, show_progress=show_progress):
            inputs = make_inputs(self.tokenizer, batch, self.device)
            with torch.no_grad(), TraceLayerDict(
                self.model,
                layers=self.get_layer_names_of_kind(kind),
                retain_output=True,
            ) as td:
                batch_answer_id_overrides = []
                for index, answer_id in answer_id_overrides:
                    if index >= total_processed and index < total_processed + len(
                        batch
                    ):
                        batch_answer_id_overrides.append(
                            (index - total_processed, answer_id)
                        )
                answer_token_ids, base_scores = predict_from_input(
                    self.model, inputs, batch_answer_id_overrides
                )
                raw_attn_masks = inputs["attention_mask"].detach().cpu()
                if move_to_cpu:
                    answer_token_ids = answer_token_ids.detach().cpu()
                    base_scores = base_scores.detach().cpu()
                answer_ids_outputs.append(answer_token_ids)
                base_scores_outputs.append(base_scores)
                for name, trace in td.items():
                    assert trace.output is not None  # keep mypy happy
                    raw_output = untuple_tensor(trace.output).detach()
                    if move_to_cpu:
                        raw_output = raw_output.cpu()
                    for input_num in range(len(batch)):
                        attn_indices = raw_attn_masks[input_num].nonzero(as_tuple=True)[
                            0
                        ]
                        uncorrupted_layer = raw_output[input_num, attn_indices]

                        uncorrupted_layer_outputs[total_processed + input_num][
                            name
                        ] = uncorrupted_layer
            total_processed += len(batch)
            for attention_mask in inputs["attention_mask"].detach().cpu():
                attention_masks.append(attention_mask)
        return BaseOutputs(
            answer_token_ids=torch.cat(answer_ids_outputs),
            base_scores=torch.cat(base_scores_outputs),
            uncorrupted_layer_outputs=uncorrupted_layer_outputs,
            attention_masks=attention_masks,
        )

    def calculate_hidden_flows(
        self,
        queries: list[str | HiddenFlowQuery],
        samples: int = 10,
        window: int = 10,
        kind: LayerKind = "hidden",
        patch_seed: int = 1,
        batch_size: int = 32,
        start_layer: int = 0,
        end_layer: Optional[int] = None,
        patch_subject_tokens_only: bool = False,
    ) -> list[HiddenFlow]:
        prompts: list[str] = []
        subjects: list[str] = []
        answer_id_overrides: list[tuple[int, int]] = []
        for i, query in enumerate(queries):
            if isinstance(query, str):
                prompts.append(query)
                subjects.append(guess_subject(query))
            else:
                prompts.append(query.text)
                subjects.append(query.subject or guess_subject(query.text))
                if query.override_answer_id is not None:
                    answer_id_overrides.append((i, query.override_answer_id))
        trace_patch_processor = AsyncTracePatchProcessor(
            self.model,
            self.tokenizer,
            self.embed_layername,
            self.noise,
            samples_per_patch=samples,
            batch_size=batch_size,
            device=self.device,
            random_seed=patch_seed,
        )
        base_outputs = self.calculate_base_outputs_batched(
            prompts, kind, batch_size, answer_id_overrides=answer_id_overrides
        )
        uncorrupted_layer_outputs = base_outputs.uncorrupted_layer_outputs

        answers = decode_tokens(self.tokenizer, base_outputs.answer_token_ids)
        inputs = make_inputs(self.tokenizer, prompts, self.device)
        subject_ranges = [
            find_token_range(self.tokenizer, input_ids, subject)
            for input_ids, subject in zip(inputs["input_ids"], subjects)
        ]
        ntoks = inputs["attention_mask"].sum(dim=1).tolist()
        hidden_flows = []
        for index, prompt in enumerate(prompts):
            subject_range = subject_ranges[index]
            token_range = subject_range if patch_subject_tokens_only else None
            answer_token_id = base_outputs.answer_token_ids[index]
            low_score_future = trace_patch_processor.trace_with_patch(
                prompt,
                states_to_patch=[],
                uncorrupted_layer_outputs=uncorrupted_layer_outputs[index],
                answer_token_id=answer_token_id,
                subject_range=subject_range,
            )
            if kind == "hidden":
                differences_futures = self.trace_important_states(
                    prompt,
                    uncorrupted_layer_outputs[index],
                    subject_range,
                    answer_token_id,
                    ntoks=ntoks[index],
                    trace_patch_processor=trace_patch_processor,
                    start_layer=start_layer,
                    end_layer=end_layer,
                    token_range=token_range,
                )
            else:
                differences_futures = self.trace_important_window(
                    prompt,
                    uncorrupted_layer_outputs[index],
                    subject_range,
                    answer_token_id,
                    window=window,
                    kind=kind,
                    trace_patch_processor=trace_patch_processor,
                    ntoks=ntoks[index],
                    start_layer=start_layer,
                    end_layer=end_layer,
                    token_range=token_range,
                )
            # need to call this to resolve all the patch trace outputs
            trace_patch_processor.process()
            low_score = low_score_future.result.item()
            differences = self.calculate_differences(
                differences_futures,
                low_score,
                start_layer=start_layer,
                end_layer=end_layer,
                ntoks=ntoks[index],
                token_range=token_range,
            )
            hidden_flows.append(
                HiddenFlow(
                    scores=differences,
                    low_score=low_score,
                    high_score=base_outputs.base_scores[index].item(),
                    input_ids=inputs["input_ids"][index],
                    input_tokens=decode_tokens(
                        self.tokenizer, inputs["input_ids"][index]
                    ),
                    subject_range=subject_range,
                    answer=answers[index],
                    kind=kind,
                    layer_outputs=detach_layer_outputs(
                        uncorrupted_layer_outputs[index]
                    ),
                )
            )
        return hidden_flows

    def calculate_differences(
        self,
        differences_futures: list[list[PseudoFuture[torch.Tensor]]],
        low_score: float,
        start_layer: int,
        end_layer: Optional[int],
        ntoks: int,
        token_range: Optional[tuple[int, int]],
    ) -> torch.Tensor:
        resolved_differences: list[torch.Tensor] = []
        for row in differences_futures:
            resolved_differences.append(torch.stack([item.result for item in row]))
        differences = torch.stack(resolved_differences).detach().cpu()

        # fill in any skipped layers with the low score
        if start_layer > 0:
            prepend_differences = torch.full(
                (differences.shape[0], start_layer), low_score
            )
            differences = torch.cat((prepend_differences, differences), dim=1)
        if end_layer is not None and end_layer < self.count_layers():
            append_differences = torch.full(
                (differences.shape[0], self.count_layers() - end_layer), low_score
            )
            differences = torch.cat((differences, append_differences), dim=1)

        if token_range is not None and token_range[0] > 0:
            prepend_differences = torch.full(
                (token_range[0], differences.shape[1]), low_score
            )
            differences = torch.cat((prepend_differences, differences), dim=0)
        if token_range is not None and token_range[1] < ntoks:
            append_differences = torch.full(
                (ntoks - token_range[1], differences.shape[1]), low_score
            )
            differences = torch.cat((differences, append_differences), dim=0)
        return differences


def detach_layer_outputs(
    layer_outputs: OrderedDict[str, torch.Tensor]
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict([(k, v.detach().cpu()) for k, v in layer_outputs.items()])
