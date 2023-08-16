import pytest
import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, LlamaTokenizer
from causal_tracer.lib.token_utils import (
    decode_tokens,
    find_final_attention_positions,
    find_final_subject_token_index,
    find_num_subject_tokens,
    get_answer_token_ids,
    predict_from_input,
    predict_logits_from_input,
    predict_next_tokens_greedy,
    predict_token,
    find_token_range,
    make_inputs,
    tokenize_with_suffix_tokens,
)


def test_decode_tokens(tokenizer: GPT2TokenizerFast) -> None:
    tokens = tokenizer.encode("Hello, my dog is cute")
    decoded = decode_tokens(tokenizer, tokens)
    assert decoded == ["Hello", ",", " my", " dog", " is", " cute"]


def test_find_token_range(tokenizer: GPT2TokenizerFast) -> None:
    tokens = tokenizer.encode("Hello, my dog is cute")
    assert find_token_range(tokenizer, tokens, "dog") == (3, 4)


def test_find_token_range_returns_the_final_instance_of_the_token_by_default(
    tokenizer: GPT2TokenizerFast,
) -> None:
    tokens = tokenizer.encode("Hello, my dog is a cute dog")
    assert find_token_range(tokenizer, tokens, "dog") == (7, 8)


def test_find_token_range_returns_the_first_instance_of_the_token_if_requested(
    tokenizer: GPT2TokenizerFast,
) -> None:
    tokens = tokenizer.encode("Hello, my dog is a cute dog")
    assert find_token_range(tokenizer, tokens, "dog", find_last_match=False) == (3, 4)


def test_find_token_range_spanning_multiple_tokens(
    tokenizer: GPT2TokenizerFast,
) -> None:
    tokens = tokenizer.encode("Hello, my dog is cute")
    assert find_token_range(tokenizer, tokens, "dog is") == (3, 5)


def test_find_token_range_non_alphanum(tokenizer: GPT2TokenizerFast) -> None:
    tokens = tokenizer.encode("Jean Pierre Joseph d’Arcet has the profession of")
    assert find_token_range(tokenizer, tokens, "Jean Pierre Joseph d’Arcet") == (0, 8)


def test_find_token_range_accented(tokenizer: GPT2TokenizerFast) -> None:
    tokens = tokenizer.encode("I think Örebro is in the continent of")
    assert find_token_range(tokenizer, tokens, "Örebro") == (2, 5)


def test_find_token_missing_token(tokenizer: GPT2TokenizerFast) -> None:
    tokens = tokenizer.encode("Hello, my dog is cute")
    with pytest.raises(ValueError):
        find_token_range(tokenizer, tokens, "cat")


def test_find_token_works_with_vicuna(vicuna_tokenizer: LlamaTokenizer) -> None:
    tokens = vicuna_tokenizer.encode("I think Paris is located in the country of")
    assert find_token_range(vicuna_tokenizer, tokens, "Paris") == (3, 4)


def test_find_final_subject_token_index(tokenizer: GPT2TokenizerFast) -> None:
    assert (
        find_final_subject_token_index(
            tokenizer, "Bill Gates is the CEO of", "Bill Gates"
        )
        == 1
    )
    assert (
        find_final_subject_token_index(tokenizer, "Bill Gates is the CEO of", "Bill")
        == 0
    )


def test_find_num_subject_tokens(tokenizer: GPT2TokenizerFast) -> None:
    assert (
        find_num_subject_tokens(tokenizer, "Bill Gates is the CEO of", "Bill Gates")
        == 2
    )
    assert find_num_subject_tokens(tokenizer, "Bill Gates is the CEO of", "Bill") == 1


def test_predict_from_input(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tokens = tokenizer("Steve Jobs is the CEO of", return_tensors="pt")
    pred, p = predict_from_input(model, tokens)
    assert p > 0
    assert p < 1
    assert tokenizer.decode(pred) == " Apple"


def test_predict_from_input_with_overrides(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    tokens = tokenizer("Steve Jobs is the CEO of", return_tensors="pt")
    original_pred, original_probs = predict_from_input(model, tokens)
    override_id = tokenizer(" Microsoft", return_tensors="pt")["input_ids"][0, 0]
    pred, probs = predict_from_input(model, tokens, [(0, override_id)])
    assert tokenizer.decode(pred) == " Microsoft"
    assert probs[0] < original_probs[0]


def test_predict_token(model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast) -> None:
    preds, probs = predict_token(
        model, tokenizer, ["Steve Jobs is the CEO of", "Bill Gates is the CEO of"]
    )
    assert probs.shape == (2,)
    assert preds == [" Apple", " Microsoft"]


def test_predict_logits_from_input(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    prompts = [
        "Steve Jobs is the CEO of",
        "Bill Gates is the CEO of",
        "Hello world",
    ]
    logits = predict_logits_from_input(model, make_inputs(tokenizer, prompts))
    assert logits.shape == (3, 50257)


def test_make_inputs(tokenizer: GPT2TokenizerFast) -> None:
    prompts = [
        "Steve Jobs is the CEO of",
        "Bill Gates is the CEO of",
        "Hello world",
    ]
    inputs = make_inputs(tokenizer, prompts)
    assert inputs["input_ids"].shape == (3, 6)
    assert inputs["attention_mask"].shape == (3, 6)
    assert torch.allclose(
        inputs["attention_mask"][0, :], torch.ones(6, dtype=torch.long)
    )
    assert torch.allclose(
        inputs["attention_mask"][1, :], torch.ones(6, dtype=torch.long)
    )
    assert torch.allclose(
        inputs["attention_mask"][2, :2], torch.ones(2, dtype=torch.long)
    )
    assert torch.allclose(
        inputs["attention_mask"][2, 2:], torch.zeros(4, dtype=torch.long)
    )


def test_find_final_attention_positions_right_padding() -> None:
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ]
    )
    assert find_final_attention_positions(attention_mask).tolist() == [2, 3]


def test_find_final_attention_positions_left_padding() -> None:
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ]
    )
    assert find_final_attention_positions(attention_mask).tolist() == [5, 5]


def test_predict_next_tokens_greedy(
    model: GPT2LMHeadModel, tokenizer: GPT2TokenizerFast
) -> None:
    prompts = [
        "Steve Jobs is the CEO of",
        "Bill Gates is the CEO of",
        "Tokyo is located in the country of",
    ]
    next_tokens = predict_next_tokens_greedy(model, tokenizer, prompts, 4)
    assert len(next_tokens) == 3
    assert next_tokens[0] == [4196, 13, 679, 318]
    assert next_tokens[1] == [5413, 11, 290, 339]
    assert next_tokens[2] == [2869, 11, 290, 318]
    assert tokenizer.decode(next_tokens[0][0]) == " Apple"
    assert tokenizer.decode(next_tokens[1][0]) == " Microsoft"
    assert tokenizer.decode(next_tokens[2][0]) == " Japan"


def test_get_answer_token_ids_adds_a_space_prefix(tokenizer: GPT2TokenizerFast) -> None:
    assert get_answer_token_ids(tokenizer, "Apple") == [4196]
    assert get_answer_token_ids(tokenizer, " Apple") == [4196]


def test_get_answer_token_ids_strips_start_tokens(
    vicuna_tokenizer: LlamaTokenizer,
) -> None:
    assert get_answer_token_ids(vicuna_tokenizer, "Apple") == [12113]
    assert get_answer_token_ids(vicuna_tokenizer, "OK Apple") == [9280, 12113]


def test_tokenize_with_suffix_tokens(tokenizer: GPT2TokenizerFast) -> None:
    PAD = tokenizer.pad_token_id
    res = tokenize_with_suffix_tokens(
        tokenizer, ["meh what?", "country", "who am I?"], [[12113], [9280, 12113], []]
    )
    assert res["input_ids"].tolist() == [
        [1326, 71, 644, 30, 12113],
        [19315, 9280, 12113, PAD, PAD],
        [8727, 716, 314, 30, PAD],
    ]
    assert res["attention_mask"].tolist() == [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
    ]
