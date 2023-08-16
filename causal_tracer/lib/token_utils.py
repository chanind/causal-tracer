# from https://github.com/kmeng01/rome/blob/main/experiments/causal_trace.py

from __future__ import annotations
from typing import Iterable, Sequence

from tokenizers import Tokenizer
from transformers import BatchEncoding
import torch
from torch import nn

from causal_tracer.lib.constants import DEFAULT_DEVICE
from causal_tracer.lib.util import find_all_substring_indices


# Utilities for dealing with tokens
def make_inputs(
    tokenizer: Tokenizer,
    prompts: Sequence[str],
    device: torch.device = DEFAULT_DEVICE,
    add_pad_token: bool = True,
) -> dict[str, torch.Tensor]:
    ensure_tokenizer_has_pad_token(tokenizer, add_pad_token=add_pad_token)
    return tokenizer(prompts, padding=True, return_tensors="pt").to(device)


def ensure_tokenizer_has_pad_token(
    tokenizer: Tokenizer, add_pad_token: bool = True
) -> None:
    # from https://github.com/huggingface/transformers/issues/12594#issuecomment-877358955
    if not tokenizer.pad_token:
        if add_pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer must have a pad token")


def decode_tokens(
    tokenizer: Tokenizer, token_array: list[int] | torch.Tensor | list[torch.Tensor]
) -> list[str]:
    return [tokenizer.decode([t]) for t in token_array]


def get_answer_token_ids(
    tokenizer: Tokenizer,
    answer: str,
    ensure_space_prefix: bool = True,
    strip_start_token: bool = True,
    strip_blank_start_token: bool = True,
) -> list[int]:
    """
    Helper to find the token ids for the given answer as if it were a continuation of the prompt.
    """
    processed_answer = answer
    if ensure_space_prefix and not processed_answer.startswith(" "):
        processed_answer = " " + processed_answer
    tokens = tokenizer.encode(processed_answer)
    if strip_start_token and tokens[0] == tokenizer.bos_token_id:
        tokens = tokens[1:]
    # llama only includes an explicit space token at the start of the string if it's the first token
    if strip_blank_start_token and tokenizer.decode([tokens[0]]) == "":
        tokens = tokens[1:]
    return tokens


def tokenize_with_suffix_tokens(
    tokenizer: Tokenizer,
    prompts: Iterable[str],
    suffix_tokens: Iterable[Iterable[int]],
) -> BatchEncoding:
    """
    Helper to tokenize a list of prompts with a list of suffix tokens appended.
    This attempts to recreate similar functionality to tokenizer.__call__(),
    but with the ability to append suffix tokens while still maintaining the correct
    attention mask. This is much less efficient than using the built-in tokenizer
    """
    ensure_tokenizer_has_pad_token(tokenizer)
    prompts_tokens = tokenizer(prompts)["input_ids"]
    combined_tokens = [
        [*prompt_tokens, *suffix]
        for prompt_tokens, suffix in zip(prompts_tokens, suffix_tokens)
    ]
    return tokenizer.pad(
        {"input_ids": combined_tokens},
        padding=True,
        return_tensors="pt",
    )


def find_token_range(
    tokenizer: Tokenizer,
    token_array: list[int] | torch.Tensor,
    substring: str,
    find_last_match: bool = True,
) -> tuple[int, int]:
    # sometimes the tokenizer messes with non-alphanumeric characters
    # so make sure the substring goes through an encoding/decoding cycle as well
    substr_toks = decode_tokens(tokenizer, tokenizer(substring)["input_ids"])
    # we want to remove the start of sentence token if the tokenizer adds it
    if tokenizer.bos_token and substr_toks[0] == tokenizer.bos_token:
        substr_toks = substr_toks[1:]
    recoded_substr = "".join(substr_toks)
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    char_locs = find_all_substring_indices(whole_string, recoded_substr)
    if len(char_locs) == 0:
        # sometimes adding a space in front causes different tokenization which works
        if substring[0] != " ":
            return find_token_range(tokenizer, token_array, " " + substring)
        raise ValueError(f"Could not find substring {recoded_substr} in {whole_string}")
    token_ranges: list[tuple[int, int]] = []
    for char_loc in char_locs:
        loc = 0
        tok_start, tok_end = None, None
        for i, t in enumerate(toks):
            loc += len(t)
            if tok_start is None and loc > char_loc:
                tok_start = i
            if tok_end is None and loc >= char_loc + len(recoded_substr):
                tok_end = i + 1
                break
        if tok_start is not None and tok_end is not None:
            token_ranges.append((tok_start, tok_end))
    if len(token_ranges) == 0:
        raise ValueError(f"Could not find substring {recoded_substr} in {toks}")
    return token_ranges[-1] if find_last_match else token_ranges[0]


def find_final_subject_token_index(
    tokenizer: Tokenizer, prompt: str, subject: str
) -> int:
    tokens = tokenizer.encode(prompt)
    _start, end = find_token_range(tokenizer, tokens, subject)
    return end - 1


def find_num_subject_tokens(tokenizer: Tokenizer, prompt: str, subject: str) -> int:
    tokens = tokenizer.encode(prompt)
    start, end = find_token_range(tokenizer, tokens, subject)
    return end - start


def predict_token(
    model: nn.Module, tokenizer: Tokenizer, prompts: list[str]
) -> tuple[list[str], torch.Tensor]:
    inp = make_inputs(tokenizer, prompts)
    preds, probs = predict_from_input(model, inp)
    results: list[str] = [tokenizer.decode(c) for c in preds]
    return (results, probs)


def predict_from_input(
    model: nn.Module,
    inp: dict[str, torch.Tensor],
    answer_id_overrides: list[tuple[int, int]] = [],
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = predict_logits_from_input(model, inp)
    probs = torch.softmax(logits, dim=-1)
    prob, pred = torch.max(probs, dim=1)
    for i, j in answer_id_overrides:
        pred[i] = j
        prob[i] = probs[i, j]
    return pred, prob


def predict_logits_from_input(
    model: nn.Module,
    inp: dict[str, torch.Tensor],
) -> torch.Tensor:
    all_logits = model(**inp)["logits"]
    final_token_positions = find_final_attention_positions(inp["attention_mask"])
    batch_indices = torch.arange(all_logits.size(0))
    return all_logits[batch_indices, final_token_positions]


def predict_next_tokens_greedy(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompts: list[str],
    num_tokens: int = 1,
    device: torch.device = DEFAULT_DEVICE,
) -> list[list[int]]:
    """
    Greedily predict the next N tokens for each prompt in the list.
    Should correctly handle right-padding.
    """
    next_prompts = [*prompts]  # copy to avoid modifying the original
    results: list[list[int]] = []
    for _i in range(num_tokens):
        # decoding and then re-encoding in a loop is wasteful, but it's the easiest way to handle
        # batches of different lengths, since model.generate() doesn't work with right-padding
        inputs = make_inputs(
            tokenizer,
            next_prompts,
            device=device,
        )
        pred_res = predict_from_input(model, inputs)
        for j, pred in enumerate(pred_res[0].detach().cpu()):
            if j >= len(results):
                results.append([])
            results[j].append(pred.item())
            next_prompts[j] += tokenizer.decode(pred)
    return results


def find_final_attention_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    # convoluted, and generated by ChatGPT, but seems to work
    indices = torch.arange(attention_mask.size(1)).to(attention_mask.device)
    # use broadcasting to expand indices to the shape of attention_mask
    indices = indices[None, :].expand_as(attention_mask)
    # set indices where attention_mask is 0 to -1
    indices = torch.where(attention_mask == 1, indices, -1)
    # find the max indices
    max_indices = indices.max(dim=1).values
    return max_indices


def ensure_space_prefix(token_str: str) -> str:
    return " " + token_str.strip()


def answer_matches_expected(answer: str, expected_answers: Iterable[str]) -> bool:
    """
    Check if the given answer matches any of the expected answers. Handles case and whitespace.
    """
    processed_answer = process_answer(answer)
    return processed_answer in {process_answer(a) for a in expected_answers}


def process_answer(answer: str) -> str:
    """
    Process the given answer to make it easier to compare to other answers by removing case and trimming.
    """
    return answer.strip().lower()
