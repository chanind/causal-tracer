from causal_tracer.lib.util import (
    find_all_substring_indices,
)


def test_find_all_substring_indices() -> None:
    assert find_all_substring_indices("Hello, World!", "l") == [2, 3, 10]
    assert find_all_substring_indices("Hello, World!", "l", 3) == [3, 10]
    assert find_all_substring_indices("Hello, World!", "l", 3, 9) == [3]
    assert find_all_substring_indices("Hello, World!", "Hello") == [0]
