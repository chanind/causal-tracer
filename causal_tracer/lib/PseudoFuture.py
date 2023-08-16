from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class PseudoFuture(Generic[T]):
    """
    A lightweight object that represents a future value.
    If the value is accessed before it's set an error is thrown.
    Just a way to have lighter-weight futures without needing asyncio.
    """

    _result: T | None = None

    def set_result(self, result: T) -> None:
        if self._result is not None:
            raise ValueError("Result is already set")
        self._result = result

    @property
    def result(self) -> T:
        if self._result is None:
            raise ValueError("Result is not set")
        return self._result
