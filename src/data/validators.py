from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.label_space import LABEL2ID, MER5_LABELS, canonicalize_vnemos_label
from src.data.schemas import (
    PAIRED_COMPAT_CSV_REQUIRED_COLUMNS,
    PAIRED_JSONL_REQUIRED_FIELDS,
    SPEECH_REQUIRED_COLUMNS,
    TEXT_REQUIRED_COLUMNS,
)


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _resolve_data_path(path_value: str, project_root: str | Path | None = None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if project_root is None:
        return path
    return Path(project_root) / path


def _assert_label_integrity(df: pd.DataFrame, name: str) -> None:
    bad_labels = sorted(set(df["label"]) - set(MER5_LABELS))
    if bad_labels:
        raise ValueError(f"{name} contains unknown labels: {bad_labels}")

    mapped = df["label"].map(LABEL2ID)
    mismatch = df[mapped != df["label_id"]]
    if not mismatch.empty:
        raise ValueError(f"{name} contains inconsistent label/label_id rows")


def validate_text_manifest(df: pd.DataFrame, name: str = "text_manifest") -> None:
    _require_columns(df, TEXT_REQUIRED_COLUMNS, name)
    _assert_label_integrity(df, name)
    if df["id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate ids")
    if df["normalized_text"].isna().any():
        raise ValueError(f"{name} contains null normalized_text values")


def validate_speech_manifest(
    df: pd.DataFrame,
    name: str = "speech_manifest",
    check_paths: bool = True,
    project_root: str | Path | None = None,
) -> None:
    _require_columns(df, SPEECH_REQUIRED_COLUMNS, name)
    _assert_label_integrity(df, name)
    if df[["path", "filename", "group_id"]].isna().any().any():
        raise ValueError(f"{name} contains null speech metadata")

    if check_paths:
        missing_paths = [
            str(_resolve_data_path(p, project_root))
            for p in df["path"].tolist()
            if not _resolve_data_path(p, project_root).exists()
        ]
        if missing_paths:
            raise ValueError(f"{name} contains missing audio paths, e.g. {missing_paths[:3]}")


def _text_signature(df: pd.DataFrame) -> pd.Series:
    joined = (
        df["normalized_text"].astype(str)
        + "||"
        + df["orig_label"].astype(str)
        + "||"
        + df["label"].astype(str)
    )
    return joined.map(lambda x: hashlib.sha1(x.encode("utf-8")).hexdigest())


def _pairwise_no_overlap(series_list: Iterable[tuple[str, pd.Series]]) -> None:
    materialized = [(name, set(series.tolist())) for name, series in series_list]
    for i, (name_a, set_a) in enumerate(materialized):
        for name_b, set_b in materialized[i + 1 :]:
            overlap = set_a & set_b
            if overlap:
                raise ValueError(
                    f"Overlap detected between {name_a} and {name_b}: {len(overlap)} rows"
                )


def validate_text_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, int]:
    validate_text_manifest(train_df, "uit_train")
    validate_text_manifest(valid_df, "uit_valid")
    validate_text_manifest(test_df, "uit_test")

    _pairwise_no_overlap(
        [
            ("train", _text_signature(train_df)),
            ("valid", _text_signature(valid_df)),
            ("test", _text_signature(test_df)),
        ]
    )

    return {
        "train": len(train_df),
        "valid": len(valid_df),
        "test": len(test_df),
    }


def validate_speech_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    project_root: str | Path | None = None,
) -> dict[str, int]:
    validate_speech_manifest(train_df, "vnemos_train", project_root=project_root)
    validate_speech_manifest(valid_df, "vnemos_valid", project_root=project_root)
    validate_speech_manifest(test_df, "vnemos_test", project_root=project_root)

    _pairwise_no_overlap(
        [
            ("train_paths", train_df["path"]),
            ("valid_paths", valid_df["path"]),
            ("test_paths", test_df["path"]),
        ]
    )

    _pairwise_no_overlap(
        [
            ("train_groups", train_df["group_id"]),
            ("valid_groups", valid_df["group_id"]),
            ("test_groups", test_df["group_id"]),
        ]
    )

    return {
        "train": len(train_df),
        "valid": len(valid_df),
        "test": len(test_df),
    }


def validate_paired_jsonl_frame(
    df: pd.DataFrame,
    name: str = "paired_manifest",
    check_paths: bool = True,
    project_root: str | Path | None = None,
) -> None:
    _require_columns(df, PAIRED_JSONL_REQUIRED_FIELDS, name)

    if df["utterance_id"].isna().any():
        raise ValueError(f"{name} contains null utterance_id values")
    if df["utterance_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate utterance_id values")

    if df["emotion"].isna().any():
        raise ValueError(f"{name} contains null emotion values")
    if df["split"].isna().any():
        raise ValueError(f"{name} contains null split values")

    allowed_splits = {"train", "valid", "test"}
    invalid_splits = sorted(set(df["split"].astype(str)) - allowed_splits)
    if invalid_splits:
        raise ValueError(f"{name} contains invalid splits: {invalid_splits}")

    canonical_emotions = [canonicalize_vnemos_label(value) for value in df["emotion"].astype(str).tolist()]
    if canonical_emotions != df["emotion"].astype(str).tolist():
        raise ValueError(f"{name} contains non-canonical emotion labels")

    transcript_final = df["transcript_final"].fillna("").astype(str)
    text_for_model = df["text_for_model"].fillna("").astype(str)

    if (transcript_final.str.strip() == "").any():
        raise ValueError(f"{name} contains empty transcript_final values")
    if (text_for_model.str.strip() == "").any():
        raise ValueError(f"{name} contains empty text_for_model values")

    mismatch = transcript_final != text_for_model
    if mismatch.any():
        raise ValueError(
            f"{name} requires text_for_model == transcript_final for MER canonical input"
        )

    if check_paths:
        missing_paths = [
            str(_resolve_data_path(p, project_root))
            for p in df["audio_path"].astype(str).tolist()
            if not _resolve_data_path(p, project_root).exists()
        ]
        if missing_paths:
            raise ValueError(
                f"{name} contains missing processed audio paths, e.g. {missing_paths[:3]}"
            )

    stems = [Path(path).stem for path in df["audio_path"].astype(str).tolist()]
    if stems != df["utterance_id"].astype(str).tolist():
        raise ValueError(f"{name} requires stem(audio_path) == utterance_id")


def validate_paired_jsonl_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    project_root: str | Path | None = None,
) -> dict[str, int]:
    validate_paired_jsonl_frame(train_df, "paired_train", project_root=project_root)
    validate_paired_jsonl_frame(valid_df, "paired_valid", project_root=project_root)
    validate_paired_jsonl_frame(test_df, "paired_test", project_root=project_root)

    _pairwise_no_overlap(
        [
            ("train_utterance_id", train_df["utterance_id"]),
            ("valid_utterance_id", valid_df["utterance_id"]),
            ("test_utterance_id", test_df["utterance_id"]),
        ]
    )

    return {
        "train": len(train_df),
        "valid": len(valid_df),
        "test": len(test_df),
    }


def validate_paired_compatibility_manifest(
    df: pd.DataFrame,
    name: str = "paired_compatibility_manifest",
    check_paths: bool = True,
    project_root: str | Path | None = None,
) -> None:
    _require_columns(df, PAIRED_COMPAT_CSV_REQUIRED_COLUMNS, name)
    _assert_label_integrity(df, name)

    if df["utterance_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate utterance_id values")
    if df["sample_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate sample_id values")
    if df["id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate id values")

    if not (df["utterance_id"].astype(str) == df["sample_id"].astype(str)).all():
        raise ValueError(f"{name} requires sample_id == utterance_id")

    if not (df["utterance_id"].astype(str) == df["id"].astype(str)).all():
        raise ValueError(f"{name} requires id == utterance_id")

    if not (df["emotion"].astype(str) == df["label"].astype(str)).all():
        raise ValueError(f"{name} requires emotion == label")

    if not (df["text_for_model"].astype(str) == df["text"].astype(str)).all():
        raise ValueError(f"{name} requires text_for_model == text")

    if check_paths:
        missing_paths = [
            str(_resolve_data_path(p, project_root))
            for p in df["audio_path"].astype(str).tolist()
            if not _resolve_data_path(p, project_root).exists()
        ]
        if missing_paths:
            raise ValueError(
                f"{name} contains missing processed audio paths, e.g. {missing_paths[:3]}"
            )

    stems = [Path(path).stem for path in df["audio_path"].astype(str).tolist()]
    if stems != df["utterance_id"].astype(str).tolist():
        raise ValueError(f"{name} requires stem(audio_path) == utterance_id")


def validate_paired_compatibility_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    project_root: str | Path | None = None,
) -> dict[str, int]:
    validate_paired_compatibility_manifest(train_df, "compat_train", project_root=project_root)
    validate_paired_compatibility_manifest(valid_df, "compat_valid", project_root=project_root)
    validate_paired_compatibility_manifest(test_df, "compat_test", project_root=project_root)

    _pairwise_no_overlap(
        [
            ("train_utterance_id", train_df["utterance_id"]),
            ("valid_utterance_id", valid_df["utterance_id"]),
            ("test_utterance_id", test_df["utterance_id"]),
        ]
    )

    return {
        "train": len(train_df),
        "valid": len(valid_df),
        "test": len(test_df),
    }