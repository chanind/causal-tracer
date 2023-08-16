from __future__ import annotations

from typing import Generator, Sequence, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def find_all_substring_indices(
    string: str, substring: str, start: int = 0, end: int | None = None
) -> list[int]:
    """
    Find all indices of a substring in a string
    """
    indices = []
    while True:
        index = string.find(substring, start, end)
        if index == -1:
            break
        indices.append(index)
        start = index + len(substring)
    return indices


def batchify(
    data: Sequence[T], batch_size: int, show_progress: bool = False
) -> Generator[Sequence[T], None, None]:
    """Generate batches from data. If show_progress is True, display a progress bar."""

    for i in tqdm(
        range(0, len(data), batch_size),
        total=(len(data) // batch_size + (len(data) % batch_size != 0)),
        disable=not show_progress,
    ):
        yield data[i : i + batch_size]
