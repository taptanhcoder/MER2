from __future__ import annotations

import random
from typing import Any

import torch


SUPPORTED_TRUNCATION_STRATEGIES = {
    "first",
    "tail",
    "head_tail",
    "random_span_train_headtail_eval",
}


def normalize_truncation_strategy(value: Any) -> str:
    strategy = str(value or "first").strip().lower().replace("-", "_")

    aliases = {
        "default": "first",
        "standard": "first",
        "longest_first": "first",
        "head": "first",
        "head_only": "first",
        "tail_only": "tail",
        "headtail": "head_tail",
        "head_tail_128": "head_tail",
        "random_span": "random_span_train_headtail_eval",
        "random_span_train": "random_span_train_headtail_eval",
        "randomspan": "random_span_train_headtail_eval",
    }
    strategy = aliases.get(strategy, strategy)

    if strategy not in SUPPORTED_TRUNCATION_STRATEGIES:
        raise ValueError(
            f"Unsupported tokenizer.truncation_strategy={strategy!r}. "
            f"Supported values: {sorted(SUPPORTED_TRUNCATION_STRATEGIES)}"
        )
    return strategy


def uses_custom_truncation(tokenizer_config: dict[str, Any]) -> bool:
    strategy = normalize_truncation_strategy(
        tokenizer_config.get("truncation_strategy", "first")
    )
    return strategy != "first"


def _special_tokens_count(tokenizer: Any) -> int:
    try:
        return len(tokenizer.build_inputs_with_special_tokens([]))
    except Exception:
        return 2


def truncate_token_ids(
    token_ids: list[int],
    max_content_length: int,
    strategy: str,
    training: bool,
) -> tuple[list[int], bool, int]:
    original_len = len(token_ids)

    if max_content_length <= 0:
        return [], original_len > 0, original_len

    if original_len <= max_content_length:
        return list(token_ids), False, original_len

    strategy = normalize_truncation_strategy(strategy)

    if strategy == "first":
        return list(token_ids[:max_content_length]), True, original_len

    if strategy == "tail":
        return list(token_ids[-max_content_length:]), True, original_len

    if strategy == "head_tail":
        head_len = max_content_length // 2
        tail_len = max_content_length - head_len
        return list(token_ids[:head_len] + token_ids[-tail_len:]), True, original_len

    if strategy == "random_span_train_headtail_eval":
        if training:
            max_start = max(0, original_len - max_content_length)
            start = random.randint(0, max_start)
            end = start + max_content_length
            return list(token_ids[start:end]), True, original_len

        head_len = max_content_length // 2
        tail_len = max_content_length - head_len
        return list(token_ids[:head_len] + token_ids[-tail_len:]), True, original_len

    raise ValueError(f"Unsupported truncation strategy: {strategy}")


def encode_text_with_strategy(
    tokenizer: Any,
    text: str,
    max_length: int,
    strategy: str,
    training: bool,
) -> tuple[list[int], list[int], int, bool]:
    content_ids = tokenizer.encode(
        str(text),
        add_special_tokens=False,
        truncation=False,
    )

    special_count = _special_tokens_count(tokenizer)
    max_content_length = max(1, int(max_length) - special_count)

    truncated_content_ids, was_truncated, original_token_len = truncate_token_ids(
        token_ids=list(content_ids),
        max_content_length=max_content_length,
        strategy=strategy,
        training=training,
    )

    input_ids = tokenizer.build_inputs_with_special_tokens(truncated_content_ids)

    if len(input_ids) > int(max_length):
        input_ids = input_ids[: int(max_length)]
        was_truncated = True

    attention_mask = [1 for _ in input_ids]
    return input_ids, attention_mask, int(original_token_len), bool(was_truncated)


def pad_encoded_rows(
    tokenizer: Any,
    encoded_rows: list[dict[str, list[int]]],
    padding: Any,
    max_length: int,
) -> dict[str, torch.Tensor]:
    pad_kwargs: dict[str, Any] = {
        "padding": padding,
        "return_tensors": "pt",
    }

    # Rows are manually truncated before padding.
    # If padding=True, do not pass max_length to avoid Transformers warning:
    # "`max_length` is ignored when padding=True and there is no truncation strategy."
    if str(padding).lower() == "max_length":
        pad_kwargs["max_length"] = int(max_length)

    return tokenizer.pad(encoded_rows, **pad_kwargs)