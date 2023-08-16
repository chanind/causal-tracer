from __future__ import annotations

import pytest
from transformers import GPT2LMHeadModel

from causal_tracer.lib.torch_utils import get_module


def test_get_module(model: GPT2LMHeadModel) -> None:
    module = get_module(model, "transformer.wte")
    assert module is not None
    assert module == model.transformer.wte


def test_get_module_numbered(model: GPT2LMHeadModel) -> None:
    module = get_module(model, "transformer.h.1.mlp")
    assert module is not None
    assert module == model.transformer.h[1].mlp


def test_get_module_nonexistent(model: GPT2LMHeadModel) -> None:
    with pytest.raises(LookupError):
        get_module(model, "nonexistent")
