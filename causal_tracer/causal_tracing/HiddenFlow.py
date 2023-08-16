from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

import torch

LayerKind = Literal["hidden", "mlp", "attn"]


@dataclass
class HiddenFlow:
    scores: torch.Tensor
    low_score: float
    high_score: float
    input_ids: torch.Tensor
    input_tokens: list[str]
    subject_range: tuple[int, int]
    answer: str
    kind: LayerKind
    layer_outputs: OrderedDict[str, torch.Tensor] = field(repr=False)
