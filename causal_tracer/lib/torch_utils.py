from __future__ import annotations
from typing import Any, Optional, TypeVar

import torch
from torch import nn


def get_module(model: nn.Module, name: str) -> nn.Module:
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)


def set_requires_grad(requires_grad: bool, *models: nn.Module) -> None:
    """
    Sets requires_grad true or false for all parameters within the
    models passed.
    """
    for model in models:
        if isinstance(model, nn.Module):
            for param in model.parameters():
                param.requires_grad = requires_grad
        elif isinstance(model, (nn.Parameter, torch.Tensor)):
            model.requires_grad = requires_grad
        else:
            raise TypeError(f"unknown type {type(model)}")


def untuple_tensor(x: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return x[0] if isinstance(x, tuple) else x


T = TypeVar("T", torch.Tensor, dict[Any, Any], list[Any], tuple[Any, ...])


def recursive_tensor_copy(
    x: T,
    clone: Optional[bool] = None,
    detach: Optional[bool] = None,
    retain_grad: Optional[bool] = None,
) -> T:
    """
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_tensor_copy(v) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_tensor_copy(v) for v in x])
    else:
        assert False, f"Unknown type {type(x)} cannot be broken into tensors."
