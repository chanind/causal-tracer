import pytest
from transformers import GPT2LMHeadModel
from causal_tracer.causal_tracing.pick_noise_level import pick_noise_level


def test_pick_noise_level(model: GPT2LMHeadModel) -> None:
    noise_level = pick_noise_level(model, "transformer.wte", std_multiplier=3)
    assert pytest.approx(noise_level, 0.01) == 0.431
