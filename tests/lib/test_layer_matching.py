from __future__ import annotations
import pytest

from transformers import GPT2LMHeadModel

from causal_tracer.lib.layer_matching import (
    collect_matching_layers,
    get_layer_name,
)


def test_collect_matching_layers_template_str(model: GPT2LMHeadModel) -> None:
    template = "transformer.h.{num}.mlp"
    matching_layers = collect_matching_layers(model, template)
    assert matching_layers == [
        "transformer.h.0.mlp",
        "transformer.h.1.mlp",
        "transformer.h.2.mlp",
        "transformer.h.3.mlp",
        "transformer.h.4.mlp",
        "transformer.h.5.mlp",
        "transformer.h.6.mlp",
        "transformer.h.7.mlp",
        "transformer.h.8.mlp",
        "transformer.h.9.mlp",
        "transformer.h.10.mlp",
        "transformer.h.11.mlp",
    ]


def test_collect_matching_layers_errors_on_invalid_template(
    model: GPT2LMHeadModel,
) -> None:
    template = "transformer.h.7.mlp"
    with pytest.raises(ValueError):
        collect_matching_layers(model, template)


def test_collect_matching_layers_callable(model: GPT2LMHeadModel) -> None:
    def matcher(_model: GPT2LMHeadModel, layer_num: int) -> str:
        return f"transformer.h.{layer_num}.mlp"

    matching_layers = collect_matching_layers(model, matcher)
    assert matching_layers == [
        "transformer.h.0.mlp",
        "transformer.h.1.mlp",
        "transformer.h.2.mlp",
        "transformer.h.3.mlp",
        "transformer.h.4.mlp",
        "transformer.h.5.mlp",
        "transformer.h.6.mlp",
        "transformer.h.7.mlp",
        "transformer.h.8.mlp",
        "transformer.h.9.mlp",
        "transformer.h.10.mlp",
        "transformer.h.11.mlp",
    ]


def test_get_layer_name(model: GPT2LMHeadModel) -> None:
    template = "transformer.h.{num}.mlp"
    assert get_layer_name(model, template, 7) == "transformer.h.7.mlp"
