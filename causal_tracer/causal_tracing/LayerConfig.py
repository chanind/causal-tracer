from dataclasses import dataclass
from typing import Any, cast

from torch import nn

from causal_tracer.lib.layer_matching import LayerMatcher


@dataclass
class LayerConfig:
    hidden_layers_matcher: LayerMatcher
    attention_layers_matcher: LayerMatcher
    mlp_layers_matcher: LayerMatcher
    embedding_layer: str


GPT2_LAYER_CONFIG = LayerConfig(
    hidden_layers_matcher="transformer.h.{num}",
    attention_layers_matcher="transformer.h.{num}.attn",
    mlp_layers_matcher="transformer.h.{num}.mlp",
    embedding_layer="transformer.wte",
)

GPTJ_LAYER_CONFIG = LayerConfig(
    hidden_layers_matcher="h.{num}",
    attention_layers_matcher="h.{num}.attn",
    mlp_layers_matcher="h.{num}.mlp",
    embedding_layer="wte",
)


LLAMA_LAYER_CONFIG = LayerConfig(
    hidden_layers_matcher="model.layers.{num}",
    attention_layers_matcher="model.layers.{num}.self_attn",
    mlp_layers_matcher="model.layers.{num}.mlp",
    embedding_layer="model.embed_tokens",
)


def get_layer_config(model: nn.Module) -> LayerConfig:
    if hasattr(model, "config"):
        config = cast(Any, model.config)
        model_type: str = config.model_type
        if model_type == "gpt2":
            return GPT2_LAYER_CONFIG
        elif model_type == "gptj":
            return GPTJ_LAYER_CONFIG
        elif model_type == "llama":
            return LLAMA_LAYER_CONFIG
    raise ValueError(
        "Unknown model type. Please provide a LayerConfig to identify relevant model layers for tracing."
    )
