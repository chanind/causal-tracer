from collections import defaultdict
from dataclasses import dataclass
from tokenizers import Tokenizer
import torch
import numpy as np
import numpy.typing as npt
from torch import nn

from causal_tracer.causal_tracing.PseudoFuture import PseudoFuture
from causal_tracer.lib.token_utils import make_inputs
from causal_tracer.lib.TraceLayerDict import TraceLayerDict
from causal_tracer.lib.torch_utils import untuple_tensor
from causal_tracer.lib.constants import DEFAULT_DEVICE


@dataclass
class ResultAccumulator:
    future: PseudoFuture[torch.Tensor]
    prompt: str
    result_parts: list[torch.Tensor]
    total_samples: int


@dataclass
class TracePatchSample:
    result_id: int
    prompt: str
    sample_num: int
    uncorrupted_layer_outputs: dict[str, torch.Tensor]
    states_to_patch: list[tuple[int, str]]
    subject_range: tuple[int, int]
    random_seed: int
    answer_token_id: torch.Tensor


class AsyncTracePatchProcessor:
    """
    Handles processing of patch traces in bulk
    """

    batch_size: int
    samples_per_patch: int
    embed_layername: str
    noise: float
    model: nn.Module
    tokenizer: Tokenizer
    results_tracker: dict[int, ResultAccumulator]
    work_queue: list[TracePatchSample]
    random_seed: int
    device: torch.device
    result_id_counter: int
    _noise_cache: dict[tuple[int, int, int], npt.NDArray[np.float64]]

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Tokenizer,
        embed_layername: str,
        noise: float,
        samples_per_patch: int = 10,
        batch_size: int = 32,
        device: torch.device = DEFAULT_DEVICE,
        random_seed: int = 1,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.samples_per_patch = samples_per_patch
        self.tokenizer = tokenizer
        self.noise = noise
        self.device = device
        self.random_seed = random_seed
        self.embed_layername = embed_layername
        self.work_queue = []
        self.results_tracker = {}
        self._noise_cache = {}
        self.result_id_counter = 0

    def trace_with_patch(
        self,
        prompt: str,
        states_to_patch: list[tuple[int, str]],
        uncorrupted_layer_outputs: dict[str, torch.Tensor],
        answer_token_id: torch.Tensor,
        subject_range: tuple[int, int],
        random_seed: int = 1,
    ) -> PseudoFuture[torch.Tensor]:
        """
        Runs a single causal trace.  Given a model and a batch input, runs the batch in inference, corrupting
        a the set of runs [1...samples_per_patch] while also restoring a set of hidden states to
        the values from the uncorrupted run.

        The argument subject_range specifies an
        be corrupted by adding Gaussian noise to the embedding for the batch
        inputs other than the first element in the batch.  Alternately,
        subsequent runs could be corrupted by simply providing different
        input tokens via the passed input batch.

        Then when running, a specified set of hidden states will be uncorrupted
        by restoring their values to the same vector that they had in the
        zeroth uncorrupted run.  This set of hidden states is listed in
        states_to_patch, by listing [(token_index, layername), ...] pairs.
        To trace the effect of just a single state, this can be just a single
        token/layer pair.  To trace the effect of restoring a set of states,
        any number of token indices and layers can be listed.
        """
        result_id = self.result_id_counter
        self.result_id_counter += 1
        for sample_num in range(self.samples_per_patch):
            self.work_queue.append(
                TracePatchSample(
                    result_id=result_id,
                    sample_num=sample_num,
                    prompt=prompt,
                    uncorrupted_layer_outputs=uncorrupted_layer_outputs,
                    states_to_patch=states_to_patch,
                    subject_range=subject_range,
                    random_seed=random_seed,
                    answer_token_id=answer_token_id,
                )
            )
        result: PseudoFuture[torch.Tensor] = PseudoFuture()
        self.results_tracker[result_id] = ResultAccumulator(
            future=result,
            prompt=prompt,
            result_parts=[],
            total_samples=self.samples_per_patch,
        )
        self._process_if_full_batch()
        return result

    def process(self) -> None:
        """
        Runs the queued traces in batches
        """
        while len(self.work_queue) > 0:
            batch = self.work_queue[: self.batch_size]
            self.work_queue = self.work_queue[self.batch_size :]
            self._process_batch(batch)

    def _process_if_full_batch(self) -> None:
        """
        Runs a batch of traces if a full batch is available
        """
        while len(self.work_queue) >= self.batch_size:
            batch = self.work_queue[: self.batch_size]
            self.work_queue = self.work_queue[self.batch_size :]
            self._process_batch(batch)

    def _assert_non_mixed_batch(self, batch: list[TracePatchSample]) -> None:
        """
        Ensures that a batch of traces is not mixed between different prompts.
        This may be supported in the future, but for now it's best to assert it's not possible.
        We need to handle resolving different length outputs / predictions and attention masks
        before that can be supported.
        """
        assert (
            len({sample.prompt for sample in batch}) == 1
        ), "Batch contains mixed prompts. This is not currently supported. call process() before adding different prompts"

    def _process_batch(self, batch: list[TracePatchSample]) -> None:
        """
        Runs a single batch of traces
        """
        self._assert_non_mixed_batch(batch)
        prompts = [sample.prompt for sample in batch]
        all_layers_in_batch: set[str] = set()
        inputs = make_inputs(self.tokenizer, prompts, self.device)
        patch_spec_per_batch_index = []
        for sample in batch:
            patch_spec = defaultdict(list)
            for token, layer in sample.states_to_patch:
                all_layers_in_batch.add(layer)
                patch_spec[layer].append(token)
            patch_spec_per_batch_index.append(patch_spec)

        # Define the model-patching rule.
        def patch_rep(x: torch.Tensor, layer: str) -> torch.Tensor:
            if layer == self.embed_layername:
                # Corrupt a range of token embeddings on batch items x
                for batch_index, sample in enumerate(batch):
                    subj_start, subj_end = sample.subject_range
                    subj_len = subj_end - subj_start
                    consistent_noise = self._get_consistent_noise(
                        sample.sample_num, subj_len, x.shape[2]
                    )
                    noise = self.noise * torch.from_numpy(consistent_noise).to(x.device)
                    x[batch_index, subj_start:subj_end] += noise
                return x
            if layer not in all_layers_in_batch:
                return x
            # If this layer is in the patch_spec, restore the uncorrupted hidden state
            # for selected tokens.
            h = untuple_tensor(x)
            for batch_index, sample in enumerate(batch):
                patch_spec = patch_spec_per_batch_index[batch_index]
                for token in patch_spec[layer]:
                    h[batch_index, token] = sample.uncorrupted_layer_outputs[layer][
                        token
                    ]
            return x

        # With the patching rules defined, run the patched model in inference.
        with torch.no_grad(), TraceLayerDict(
            self.model,
            [self.embed_layername] + list(all_layers_in_batch),
            edit_output=patch_rep,
        ):
            outputs_exp = self.model(**inputs)

        for batch_index, sample in enumerate(batch):
            result_tracker = self.results_tracker[sample.result_id]
            result_tracker.result_parts.append(outputs_exp.logits[batch_index])
            # if we have all the parts of the result, compute the final result and set the future
            if len(result_tracker.result_parts) == result_tracker.total_samples:
                result_logits = torch.stack(result_tracker.result_parts)
                probs = torch.softmax(result_logits[:, -1, :], dim=1).mean(dim=0)[
                    sample.answer_token_id
                ]
                result_tracker.future.set_result(probs)
                self.results_tracker.pop(sample.result_id)

    def _get_consistent_noise(
        self, sample_num: int, subj_len: int, layer_size: int
    ) -> npt.NDArray[np.float64]:
        """
        Returns a numpy array of noise for a given sample number and shape, which is consistent for each combination of inputs
        """
        cache_key = (sample_num, subj_len, layer_size)
        if cache_key not in self._noise_cache:
            prng = np.random.RandomState(self.random_seed)
            self._noise_cache[cache_key] = prng.randn(
                self.samples_per_patch, subj_len, layer_size
            )[sample_num]
        return self._noise_cache[cache_key]
