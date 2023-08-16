from __future__ import annotations

import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

from causal_tracer.lib.TraceLayerDict import TraceLayerDict


def test_TraceLayerDict_can_spy_on_multiple_layer_outputs(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    with torch.no_grad():
        embeddings = model.transformer.wte(inputs["input_ids"])
        with TraceLayerDict(
            model, ["transformer.wte", "transformer.h.0"]
        ) as trace_dict:
            assert list(trace_dict.keys()) == ["transformer.wte", "transformer.h.0"]
            assert trace_dict["transformer.wte"].output is None
            assert trace_dict["transformer.h.0"].output is None
            model(inputs["input_ids"])
            assert trace_dict["transformer.wte"].output is not None
            assert trace_dict["transformer.h.0"].output is not None
            assert torch.allclose(trace_dict["transformer.wte"].output, embeddings)
