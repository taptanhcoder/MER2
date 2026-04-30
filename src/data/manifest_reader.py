from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.label_space import canonicalize_vnemos_label, label_to_id
from src.data.schemas import PAIRED_JSONL_WRITE_FIELDS
from src.utils.io import read_jsonl


@dataclass(frozen=True)
class PreparedPairedDataset:
    all_frame: pd.DataFrame
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    test_frame: pd.DataFrame

    compat_all_frame: pd.DataFrame
    compat_train_frame: pd.DataFrame
    compat_valid_frame: pd.DataFrame
    compat_test_frame: pd.DataFrame

    stats: dict[str, Any]


_SEGMENT_SUFFIX_PATTERN = re.compile(r"(?:[_\-\s])seg\d+$", flags=re.IGNORECASE)


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text))
    value = value.strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def infer_source_group_from_utterance_id(utterance_id: str) -> str:
    value = str(utterance_id).strip().lower()
    value = _SEGMENT_SUFFIX_PATTERN.sub("", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def resolve_project_relative_audio_path(
    utterance_id: str,
    processed_audio_root: str | Path,
) -> str:
    return str((Path(processed_audio_root) / f"{utterance_id}.wav").as_posix())


def _duration_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "num_samples": 0,
            "mean_sec": 0.0,
            "min_sec": 0.0,
            "max_sec": 0.0,
        }

    series = pd.Series(values, dtype="float64")
    return {
        "num_samples": int(series.shape[0]),
        "mean_sec": float(series.mean()),
        "min_sec": float(series.min()),
        "max_sec": float(series.max()),
    }


def load_existing_paired_splits(dataset_cfg: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    source_splits = dict(dataset_cfg["source_splits"])
    required_split_names = ["train", "valid", "test"]

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name in required_split_names:
        if split_name not in source_splits:
            raise ValueError(
                f"Missing source split path for '{split_name}' in dataset.source_splits"
            )

        split_path = Path(source_splits[split_name])
        if not split_path.exists():
            raise FileNotFoundError(f"Source split JSONL not found: {split_path}")

        rows = read_jsonl(split_path)
        rows_by_split[split_name] = rows

    return rows_by_split


def canonicalize_paired_record(
    record: dict[str, Any],
    split_name: str,
    dataset_cfg: dict[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    strict_cfg = dict(dataset_cfg.get("strict", {}))
    require_non_empty_transcript = bool(strict_cfg.get("require_non_empty_transcript", True))

    utterance_id = str(record.get("utterance_id", "")).strip()
    if not utterance_id:
        raise ValueError(f"[{split_name}] Missing utterance_id: {record}")

    raw_emotion = str(record.get("emotion", "")).strip()
    if not raw_emotion:
        raise ValueError(f"[{split_name}] Missing emotion for utterance_id={utterance_id}")

    emotion = canonicalize_vnemos_label(raw_emotion)

    transcript_final = str(record.get("transcript_final", "")).strip()
    if require_non_empty_transcript and not transcript_final:
        raise ValueError(
            f"[{split_name}] Empty transcript_final for utterance_id={utterance_id}"
        )

    text_for_model = str(record.get("text_for_model") or transcript_final).strip()
    if require_non_empty_transcript and not text_for_model:
        raise ValueError(
            f"[{split_name}] Empty text_for_model for utterance_id={utterance_id}"
        )

    declared_split = str(record.get("split", "")).strip()
    if declared_split and declared_split != split_name:
        raise ValueError(
            f"[{split_name}] Split mismatch for utterance_id={utterance_id}: "
            f"record split='{declared_split}'"
        )

    canonical_audio_path = resolve_project_relative_audio_path(
        utterance_id=utterance_id,
        processed_audio_root=dataset_cfg["processed_audio_root"],
    )

    duration_value = record.get("duration", 0.0)
    try:
        duration = float(duration_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[{split_name}] Invalid duration for utterance_id={utterance_id}: {duration_value}"
        ) from exc

    source_group = str(
        record.get("source_group")
        or record.get("group_id")
        or infer_source_group_from_utterance_id(utterance_id)
    ).strip()

    return {
        "utterance_id": utterance_id,
        "emotion": emotion,
        "orig_emotion": raw_emotion,
        "audio_path": canonical_audio_path,
        "duration": duration,
        "transcript_final": transcript_final,
        "text_for_model": text_for_model,
        "split": split_name,
        "source_group": source_group,
        "group_id": source_group,
        "filename": Path(canonical_audio_path).name,
        "project_root": str(project_root),
    }


def _canonicalize_split_rows(
    rows: list[dict[str, Any]],
    split_name: str,
    dataset_cfg: dict[str, Any],
    project_root: str | Path,
) -> pd.DataFrame:
    canonical_rows = [
        canonicalize_paired_record(
            record=row,
            split_name=split_name,
            dataset_cfg=dataset_cfg,
            project_root=project_root,
        )
        for row in rows
    ]
    return pd.DataFrame(canonical_rows)


def canonicalize_paired_splits(
    rows_by_split: dict[str, list[dict[str, Any]]],
    dataset_cfg: dict[str, Any],
    project_root: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = _canonicalize_split_rows(
        rows=rows_by_split["train"],
        split_name="train",
        dataset_cfg=dataset_cfg,
        project_root=project_root,
    )
    valid_df = _canonicalize_split_rows(
        rows=rows_by_split["valid"],
        split_name="valid",
        dataset_cfg=dataset_cfg,
        project_root=project_root,
    )
    test_df = _canonicalize_split_rows(
        rows=rows_by_split["test"],
        split_name="test",
        dataset_cfg=dataset_cfg,
        project_root=project_root,
    )
    return train_df, valid_df, test_df


def _to_jsonl_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame[PAIRED_JSONL_WRITE_FIELDS]
        .sort_values(["split", "emotion", "utterance_id"])
        .reset_index(drop=True)
    )


def _to_compatibility_frame(frame: pd.DataFrame) -> pd.DataFrame:
    compat = pd.DataFrame(
        {
            "utterance_id": frame["utterance_id"].astype(str),
            "sample_id": frame["utterance_id"].astype(str),
            "id": frame["utterance_id"].astype(str),
            "emotion": frame["emotion"].astype(str),
            "orig_label": frame["orig_emotion"].astype(str),
            "label": frame["emotion"].astype(str),
            "label_id": frame["emotion"].map(label_to_id).astype(int),
            "audio_path": frame["audio_path"].astype(str),
            "path": frame["audio_path"].astype(str),
            "filename": frame["filename"].astype(str),
            "source_group": frame["source_group"].astype(str),
            "group_id": frame["group_id"].astype(str),
            "duration": frame["duration"].astype(float),
            "transcript_final": frame["transcript_final"].astype(str),
            "text_for_model": frame["text_for_model"].astype(str),
            "text": frame["text_for_model"].astype(str),
            "normalized_text": frame["text_for_model"].map(normalize_text),
            "split": frame["split"].astype(str),
        }
    )
    return compat.sort_values(["split", "label", "utterance_id"]).reset_index(drop=True)


def _build_stats(
    all_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    valid_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    dataset_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset": str(dataset_cfg.get("name", "vnemos_paired_mer")),
        "source_mode": "existing_paired_splits",
        "source_split_paths": dict(dataset_cfg["source_splits"]),
        "processed_audio_root": str(dataset_cfg["processed_audio_root"]),
        "split_sizes": {
            "train": int(len(train_frame)),
            "valid": int(len(valid_frame)),
            "test": int(len(test_frame)),
            "all": int(len(all_frame)),
        },
        "class_distribution": {
            "train": train_frame["emotion"].value_counts().sort_index().to_dict(),
            "valid": valid_frame["emotion"].value_counts().sort_index().to_dict(),
            "test": test_frame["emotion"].value_counts().sort_index().to_dict(),
        },
        "duration_sec": {
            "all": _duration_summary(all_frame["duration"].astype(float).tolist()),
            "train": _duration_summary(train_frame["duration"].astype(float).tolist()),
            "valid": _duration_summary(valid_frame["duration"].astype(float).tolist()),
            "test": _duration_summary(test_frame["duration"].astype(float).tolist()),
        },
    }


def prepare_vnemos_paired_from_existing_splits(
    config: dict[str, Any],
) -> PreparedPairedDataset:
    project_root = Path(config.get("project", {}).get("root", ".")).resolve()
    dataset_cfg = dict(config["dataset"])

    rows_by_split = load_existing_paired_splits(dataset_cfg)
    train_src, valid_src, test_src = canonicalize_paired_splits(
        rows_by_split=rows_by_split,
        dataset_cfg=dataset_cfg,
        project_root=project_root,
    )

    all_src = pd.concat([train_src, valid_src, test_src], ignore_index=True)

    all_frame = _to_jsonl_frame(all_src)
    train_frame = _to_jsonl_frame(train_src)
    valid_frame = _to_jsonl_frame(valid_src)
    test_frame = _to_jsonl_frame(test_src)

    compat_all = _to_compatibility_frame(all_src)
    compat_train = _to_compatibility_frame(train_src)
    compat_valid = _to_compatibility_frame(valid_src)
    compat_test = _to_compatibility_frame(test_src)

    stats = _build_stats(
        all_frame=all_frame,
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        dataset_cfg=dataset_cfg,
    )

    return PreparedPairedDataset(
        all_frame=all_frame,
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        compat_all_frame=compat_all,
        compat_train_frame=compat_train,
        compat_valid_frame=compat_valid,
        compat_test_frame=compat_test,
        stats=stats,
    )