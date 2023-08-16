from __future__ import annotations

import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

from causal_tracer.lib.TraceLayer import TraceLayer


def test_Trace_can_spy_on_layer_outputs(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    with torch.no_grad():
        embeddings = model.transformer.wte(inputs["input_ids"])
        with TraceLayer(model, "transformer.wte") as trace:
            model(inputs["input_ids"])
            assert trace.output is not None
            assert torch.allclose(trace.output, embeddings)


def test_Trace_can_edit_layer_outputs(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    target_shape = inputs["input_ids"].shape + (model.transformer.wte.weight.shape[-1],)
    new_output = torch.randn(target_shape)
    with torch.no_grad():
        with TraceLayer(
            model, "transformer.wte", edit_output=lambda _output, _layer: new_output
        ) as trace:
            model(inputs["input_ids"])
            assert trace.output is not None
            assert torch.allclose(trace.output, new_output)


def test_Trace_can_edit_layer_outputs_with_args(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    target_shape = inputs["input_ids"].shape + (model.transformer.wte.weight.shape[-1],)
    new_output = torch.randn(target_shape)
    called_layer: str | None = None
    called_output: torch.Tensor | None = None

    def edit_output(output: torch.Tensor, layer: str) -> torch.Tensor:
        nonlocal called_layer, called_output
        called_layer = layer
        called_output = output
        return new_output

    with torch.no_grad():
        embeddings = model.transformer.wte(inputs["input_ids"])

        with TraceLayer(model, "transformer.wte", edit_output=edit_output) as trace:
            model(inputs["input_ids"])
            assert trace.output is not None
            assert called_output is not None
            assert torch.allclose(trace.output, new_output)
            assert called_layer == "transformer.wte"
            assert torch.allclose(called_output, embeddings)
