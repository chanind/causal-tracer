import pytest
from causal_tracer.causal_tracing.guess_subject import guess_subject


def test_guess_subject() -> None:
    assert guess_subject("Bill Gates is the CEO of Microsoft") == "Bill Gates"


def test_guess_subject_errors_on_failure() -> None:
    with pytest.raises(ValueError):
        guess_subject("234235492343")
