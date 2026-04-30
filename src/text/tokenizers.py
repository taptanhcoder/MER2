from __future__ import annotations

from typing import Any

from transformers import AutoTokenizer


def build_text_tokenizer(tokenizer_config: dict[str, Any], model_config: dict[str, Any]):
    pretrained_name = model_config["pretrained_name"]
    use_fast = tokenizer_config.get("use_fast", False)
    return AutoTokenizer.from_pretrained(pretrained_name, use_fast=use_fast)