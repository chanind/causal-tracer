"""
TraceDict will hook multiple layers at once.

Copied from https://github.com/kmeng01/rome/blob/main/util/nethook.py
"""

from __future__ import annotations

import contextlib
from collections import OrderedDict
from typing import Any, Callable, Iterable, Optional

from torch import nn

from causal_tracer.lib.TraceLayer import StopForward, TraceLayer


class TraceLayerDict(
    OrderedDict[str, TraceLayer],
    contextlib.AbstractContextManager["TraceLayerDict"],
):
    """
    To retain the output of multiple named layers during the computation
    of the given network:

        with TraceDict(net, ['layer1.name1', 'layer2.name2']) as ret:
            _ = net(inp)
            representation = ret['layer1.name1'].output

    If edit_output is provided, it should be a function that takes
    two arguments: output, and the layer name; and then it returns the
    modified output.

    Other arguments are the same as Trace.  If stop is True, then the
    execution of the network will be stopped after the last layer
    listed (even if it would not have been the last to be executed).
    """

    def __init__(
        self,
        module: nn.Module,
        layers: Optional[Iterable[str]] = None,
        retain_output: bool = True,
        retain_input: bool = False,
        clone: bool = False,
        detach: bool = False,
        retain_grad: bool = False,
        edit_output: Optional[Callable[[Any, str], Any]] = None,
        stop: bool = False,
    ):
        self.stop = stop

        def flag_last_unseen(it: Any) -> Any:
            try:
                it = iter(it)
                prev = next(it)
                seen = set([prev])
            except StopIteration:
                return
            for item in it:
                if item not in seen:
                    yield False, prev
                    seen.add(item)
                    prev = item
            yield True, prev

        for is_last, layer in flag_last_unseen(layers):
            self[layer] = TraceLayer(
                module=module,
                layer=layer,
                retain_output=retain_output,
                retain_input=retain_input,
                clone=clone,
                detach=detach,
                retain_grad=retain_grad,
                edit_output=edit_output,
                stop=stop and is_last,
            )

    def __enter__(self) -> "TraceLayerDict":
        return self

    def __exit__(self, type: Any, _value: Any, _traceback: Any) -> bool | None:
        self.close()
        if self.stop and issubclass(type, StopForward):
            return True
        return None

    def close(self) -> None:
        for _layer, trace in reversed(self.items()):
            trace.close()
