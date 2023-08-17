from typing import cast
import torch
from torch import nn
from causal_tracer.lib.torch_utils import get_module


def pick_noise_level(
    model: nn.Module,
    embedding_layer: str,
    std_multiplier: float = 3,
) -> float:
    """
    Pick a noise level to corrupt the input text with, such that the
    noise is a multiplier of the stdev of the token embeddings.
    """
    with torch.no_grad():
        embedding_weights = cast(
            torch.Tensor, get_module(model, embedding_layer).weight
        )
        noise_level_std = embedding_weights.std().item()
    return std_multiplier * noise_level_std
