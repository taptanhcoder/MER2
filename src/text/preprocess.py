from __future__ import annotations

import re
import unicodedata

try:
    from underthesea import word_tokenize
except ImportError:
    word_tokenize = None


_FILLER_PATTERN = re.compile(
    r"\b(ờ+|à+|ừ+|ừm+|um+|uh+|ah+|ờm+|hmm+)\b",
    flags=re.IGNORECASE,
)
_MULTI_DOT_PATTERN = re.compile(r"[.]{2,}")
_MULTI_Q_PATTERN = re.compile(r"[?]{2,}")
_MULTI_E_PATTERN = re.compile(r"[!]{2,}")


def normalize_text_for_training(
    text: str,
    normalize_whitespace: bool = True,
    lowercase: bool = False,
    preserve_spoken_style: bool = True,
) -> str:
    value = unicodedata.normalize("NFKC", str(text))

    if not preserve_spoken_style:
        value = _FILLER_PATTERN.sub(" ", value)

    value = _MULTI_DOT_PATTERN.sub(".", value)
    value = _MULTI_Q_PATTERN.sub("?", value)
    value = _MULTI_E_PATTERN.sub("!", value)

    if normalize_whitespace:
        value = re.sub(r"\s+", " ", value).strip()

    if lowercase:
        value = value.lower()

    return value


def word_segment_text(text: str, enabled: bool = False) -> str:
    if not enabled:
        return text
    if word_tokenize is None:
        return text
    try:
        return word_tokenize(text, format="text")
    except Exception:
        return text


def prepare_model_text(
    text: str,
    normalize_whitespace: bool = True,
    lowercase: bool = False,
    word_segment: bool = False,
    preserve_spoken_style: bool = True,
) -> str:
    value = normalize_text_for_training(
        text=text,
        normalize_whitespace=normalize_whitespace,
        lowercase=lowercase,
        preserve_spoken_style=preserve_spoken_style,
    )
    value = word_segment_text(value, enabled=word_segment)
    return value