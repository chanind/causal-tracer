import pytest
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

from causal_tracer.causal_tracing.CausalTracer import CausalTracer


def test_CausalTracer_count_layers(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tracer = CausalTracer(model, tokenizer)
    assert tracer.count_layers() == 12


def test_CausalTracer_calculate_hidden_flow_with_no_subject(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tracer = CausalTracer(model, tokenizer)
    res = tracer.calculate_hidden_flow("Bill Gates is the CEO of")
    assert res.answer == " Microsoft"
    assert res.subject_range == (0, 2)
    assert res.high_score > res.low_score
    max_scores_by_token = res.scores.max(dim=1).values
    # the final hidden state has the highest score, followed by the last subject token
    assert max_scores_by_token.topk(2, dim=0).indices.tolist() == [5, 1]
    # empirically, this is the sum. If we change the model or random seed, this will change
    assert pytest.approx(res.scores.sum(), 0.01) == 3.6942


def test_CausalTracer_calculate_hidden_flow_overwriting_answer(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tracer = CausalTracer(model, tokenizer)
    res = tracer.calculate_hidden_flow(
        "Bill Gates is the CEO of",
        answer_id_override=tokenizer(" Apple")["input_ids"][0],
    )
    assert res.answer == " Apple"
    assert res.subject_range == (0, 2)


def test_CausalTracer_calculate_hidden_flow_mlp(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tracer = CausalTracer(model, tokenizer)
    res = tracer.calculate_hidden_flow("Steve Jobs is the CEO of", kind="mlp")
    assert res.answer == " Apple"
    assert res.subject_range == (0, 2)
    assert res.high_score > res.low_score
    max_scores_by_token = res.scores.max(dim=1).values
    # The last subject token should have the highest score for the MLP layer
    assert max_scores_by_token.max(dim=0).indices == 1
    # layer 5 should have the highest score for the MLP layer for gpt2, empirically
    assert res.scores[1].max(dim=0).indices == 5
    # empirically, this is the sum. If we change the model or random seed, this will change
    assert pytest.approx(res.scores.sum(), 0.01) == 1.5386
    assert res.scores.shape == (6, 12)


def test_CausalTracer_calculate_hidden_flow_fills_in_skipped_layers_with_low_score(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tracer = CausalTracer(model, tokenizer)
    res = tracer.calculate_hidden_flow(
        "Steve Jobs is the CEO of",
        kind="mlp",
        start_layer=3,
        end_layer=7,
        samples=10,
    )
    assert res.answer == " Apple"
    assert res.subject_range == (0, 2)
    assert res.high_score > res.low_score
    max_scores_by_token = res.scores.max(dim=1).values
    # The last subject token should have the highest score for the MLP layer
    assert max_scores_by_token.max(dim=0).indices == 1
    # layer 5 should have the highest score for the MLP layer for gpt2, empirically
    assert res.scores[1].max(dim=0).indices == 5
    assert res.scores.shape == (6, 12)
    assert torch.all(res.scores[:, 0:3] == res.low_score)
    assert torch.all(res.scores[:, 7:] == res.low_score)


def test_CausalTracer_calculate_hidden_flow_fills_in_skipped_tokens_with_low_score(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tracer = CausalTracer(model, tokenizer)
    res = tracer.calculate_hidden_flow(
        "I hear Steve Jobs is the CEO of",
        subject="Steve Jobs",
        kind="mlp",
        samples=20,
        patch_subject_tokens_only=True,
    )
    assert res.answer == " Apple"
    assert res.subject_range == (2, 4)
    assert res.high_score > res.low_score
    max_scores_by_token = res.scores.max(dim=1).values
    # The last subject token should have the highest score for the MLP layer
    assert max_scores_by_token.max(dim=0).indices == 3
    assert res.scores.shape == (8, 12)
    assert torch.all(res.scores[0:2, :] == res.low_score)
    assert torch.all(res.scores[4:, :] == res.low_score)


def test_CausalTracer_calculate_hidden_flow_mlp_long_subject(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tracer = CausalTracer(model, tokenizer)
    res = tracer.calculate_hidden_flow(
        "Hiroshima is located in the country of", subject="Hiroshima", kind="mlp"
    )
    assert res.answer == " Japan"
    assert res.subject_range == (0, 4)
    assert res.high_score > res.low_score
    max_scores_by_token = res.scores.max(dim=1).values
    # The last subject token should have the highest score for the MLP layer
    assert max_scores_by_token.max(dim=0).indices == 3
    # layer 5 should have the highest score for the MLP layer for gpt2, empirically
    assert res.scores[1].max(dim=0).indices == 5
    # empirically, this is the sum. If we change the model or random seed, this will change
    assert pytest.approx(res.scores.sum(), 0.01) == 0.9466


def test_CausalTracer_calculate_hidden_flows_mlp(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tracer = CausalTracer(model, tokenizer)
    results = tracer.calculate_hidden_flows(
        [
            "Steve Jobs is the CEO of",
            "Bill Gates is the CEO of",
            "Hiroshima is located in the country of",
        ],
        kind="mlp",
    )
    assert results[0].answer == " Apple"
    assert results[1].answer == " Microsoft"
    assert results[2].answer == " Japan"
    assert results[0].subject_range == (0, 2)
    assert results[1].subject_range == (0, 2)
    assert results[2].subject_range == (0, 4)
    # empirically, this is the sum. If we change the model or random seed, this will change
    assert pytest.approx(results[0].scores.sum(), 0.01) == 1.5386
    assert pytest.approx(results[1].scores.sum(), 0.01) == 1.6196
    assert pytest.approx(results[2].scores.sum(), 0.01) == 0.9466


def test_CausalTracer_calculate_hidden_flows_small_batch_size(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tracer = CausalTracer(model, tokenizer)
    results = tracer.calculate_hidden_flows(
        [
            "Steve Jobs is the CEO of",
            "Bill Gates is the CEO of",
            "Hiroshima is located in the country of",
        ],
        kind="mlp",
        batch_size=3,
    )
    assert results[0].answer == " Apple"
    assert results[1].answer == " Microsoft"
    assert results[2].answer == " Japan"
    assert results[0].subject_range == (0, 2)
    assert results[1].subject_range == (0, 2)
    assert results[2].subject_range == (0, 4)
    # empirically, this is the sum. If we change the model or random seed, this will change
    assert pytest.approx(results[0].scores.sum(), 0.01) == 1.5386
    assert pytest.approx(results[1].scores.sum(), 0.01) == 1.6196
    assert pytest.approx(results[2].scores.sum(), 0.01) == 0.9466


def test_CausalTracer_calculate_hidden_flows_hidden(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tracer = CausalTracer(model, tokenizer)
    results = tracer.calculate_hidden_flows(
        [
            "Steve Jobs is the CEO of",
            "Bill Gates is the CEO of",
            "Hiroshima is located in the country of",
        ],
        kind="hidden",
    )
    assert results[0].answer == " Apple"
    assert results[1].answer == " Microsoft"
    assert results[2].answer == " Japan"
    assert results[0].subject_range == (0, 2)
    assert results[1].subject_range == (0, 2)
    assert results[2].subject_range == (0, 4)
    # empirically, this is the sum. If we change the model or random seed, this will change
    assert pytest.approx(results[0].scores.sum(), 0.01) == 3.7863
    assert pytest.approx(results[1].scores.sum(), 0.01) == 3.6942
    assert pytest.approx(results[2].scores.sum(), 0.01) == 1.3727


def test_CausalTracer_calculate_base_outputs_batched(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tracer = CausalTracer(model, tokenizer)
    results = tracer.calculate_base_outputs_batched(
        [
            "Steve Jobs is the CEO of",
            "Bill Gates is the CEO of",
            "Hiroshima is located in the country of",
        ],
        kind="mlp",
        batch_size=64,
    )
    assert results.base_scores.shape == (3,)
    assert len(results.attention_masks) == 3
    assert results.attention_masks[0].sum() == 6
    assert results.attention_masks[1].sum() == 6
    assert results.attention_masks[2].sum() == 10
