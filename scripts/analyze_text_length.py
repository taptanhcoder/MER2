from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.text.tokenizers import build_text_tokenizer
from src.utils.config import deep_update, load_yaml, resolve_config_reference
from src.utils.io import ensure_dir, write_csv, write_json
from src.utils.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze VNEMOS transcript length and tokenizer truncation risk "
            "for text emotion recognition configs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/text/experiments/phobert_base_ce.yaml"),
        help="Text experiment config used to resolve dataset/model/tokenizer settings.",
    )
    parser.add_argument(
        "--max-lengths",
        type=int,
        nargs="+",
        default=[128, 192],
        help="Tokenizer max_length values to audit for truncation.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("outputs/reports/data/text_length"),
        help="Directory where CSV/JSON reports will be written.",
    )
    parser.add_argument(
        "--use-normalized-text",
        action="store_true",
        help="Use dataset.normalized_text_col if available instead of dataset.text_col.",
    )
    return parser


def resolve_experiment_config(config_path: Path) -> dict[str, Any]:
    exp_cfg = load_yaml(resolve_config_reference(config_path, PROJECT_ROOT))
    defaults = exp_cfg.get("defaults", [])
    merged: dict[str, Any] = {}

    for entry in defaults:
        if not isinstance(entry, str):
            raise TypeError("Text config `defaults` entries must be string paths.")
        cfg_part = load_yaml(resolve_config_reference(entry, PROJECT_ROOT))
        merged = deep_update(merged, cfg_part)

    exp_without_defaults = dict(exp_cfg)
    exp_without_defaults.pop("defaults", None)
    merged = deep_update(merged, exp_without_defaults)
    return merged


def _read_split_frame(
    csv_path: str | Path,
    split: str,
    project_root: str | Path,
) -> pd.DataFrame:
    path = resolve_project_path(csv_path, start=project_root)
    if not path.exists():
        raise FileNotFoundError(f"{split} CSV not found: {path}")

    frame = pd.read_csv(path)
    frame["split"] = split
    return frame


def _resolve_text_column(
    dataset_cfg: dict[str, Any],
    frame: pd.DataFrame,
    use_normalized_text: bool,
) -> str:
    if use_normalized_text:
        normalized_col = str(dataset_cfg.get("normalized_text_col", "normalized_text"))
        if normalized_col in frame.columns:
            return normalized_col

    text_col = str(dataset_cfg.get("text_col", "text"))
    if text_col in frame.columns:
        return text_col

    raw_col = str(dataset_cfg.get("raw_text_col", "transcript_final"))
    if raw_col in frame.columns:
        return raw_col

    raise KeyError(
        "Could not resolve text column. "
        f"Tried text_col={text_col!r}, raw_text_col={raw_col!r}."
    )


def _safe_word_count(text: str) -> int:
    stripped = str(text).strip()
    if not stripped:
        return 0
    return len(stripped.split())


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        str(text),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    input_ids = encoded.get("input_ids", [])
    return int(len(input_ids))


def _build_per_sample_report(
    config: dict[str, Any],
    max_lengths: list[int],
    use_normalized_text: bool,
) -> pd.DataFrame:
    dataset_cfg = dict(config["dataset"])
    project_root = config.get("project", {}).get("root", PROJECT_ROOT)

    frames = [
        _read_split_frame(dataset_cfg["train_csv"], "train", project_root),
        _read_split_frame(dataset_cfg["valid_csv"], "valid", project_root),
        _read_split_frame(dataset_cfg["test_csv"], "test", project_root),
    ]
    data = pd.concat(frames, ignore_index=True)

    text_col = _resolve_text_column(dataset_cfg, data, use_normalized_text)
    label_col = str(dataset_cfg.get("label_col", "label"))
    label_id_col = str(dataset_cfg.get("label_id_col", "label_id"))
    id_col = str(dataset_cfg.get("id_col", "sample_id"))
    duration_col = str(dataset_cfg.get("duration_col", "duration"))

    tokenizer = build_text_tokenizer(
        tokenizer_config=config["tokenizer"],
        model_config=config["model"],
    )

    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        text = str(row.get(text_col, ""))
        token_len = _token_count(tokenizer, text)

        record: dict[str, Any] = {
            "split": str(row.get("split", "")),
            "sample_id": str(row.get(id_col, "")),
            "label": str(row.get(label_col, "")),
            "label_id": int(row[label_id_col]) if label_id_col in row and pd.notna(row[label_id_col]) else None,
            "text_col": text_col,
            "char_len": int(len(text)),
            "word_count": int(_safe_word_count(text)),
            "token_len": int(token_len),
            "text": text,
        }

        if duration_col in data.columns and pd.notna(row.get(duration_col)):
            record["duration"] = float(row[duration_col])

        for max_length in max_lengths:
            record[f"truncated_at_{max_length}"] = bool(token_len > int(max_length))
            record[f"overflow_tokens_at_{max_length}"] = max(0, int(token_len) - int(max_length))

        rows.append(record)

    return pd.DataFrame(rows)


def _summarize_by_split_label(
    per_sample_df: pd.DataFrame,
    max_lengths: list[int],
) -> pd.DataFrame:
    numeric_aggs: dict[str, list[str]] = {
        "sample_id": ["count"],
        "char_len": ["mean", "std", "min", "max"],
        "word_count": ["mean", "std", "min", "max"],
        "token_len": ["mean", "std", "min", "max"],
    }

    if "duration" in per_sample_df.columns:
        numeric_aggs["duration"] = ["mean", "std", "min", "max"]

    for max_length in max_lengths:
        numeric_aggs[f"truncated_at_{max_length}"] = ["mean", "sum"]
        numeric_aggs[f"overflow_tokens_at_{max_length}"] = ["mean", "max"]

    grouped = (
        per_sample_df.groupby(["split", "label"], dropna=False)
        .agg(numeric_aggs)
        .reset_index()
    )

    grouped.columns = [
        "__".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]

    rename_map = {
        "sample_id__count": "num_samples",
        "split__": "split",
        "label__": "label",
    }
    grouped = grouped.rename(columns=rename_map)

    for max_length in max_lengths:
        rate_col = f"truncated_at_{max_length}__mean"
        sum_col = f"truncated_at_{max_length}__sum"
        if rate_col in grouped.columns:
            grouped = grouped.rename(columns={rate_col: f"truncation_rate_at_{max_length}"})
        if sum_col in grouped.columns:
            grouped = grouped.rename(columns={sum_col: f"num_truncated_at_{max_length}"})

    return grouped


def _summarize_by_label(
    per_sample_df: pd.DataFrame,
    max_lengths: list[int],
) -> pd.DataFrame:
    numeric_aggs: dict[str, list[str]] = {
        "sample_id": ["count"],
        "char_len": ["mean", "std", "min", "max"],
        "word_count": ["mean", "std", "min", "max"],
        "token_len": ["mean", "std", "min", "max"],
    }

    if "duration" in per_sample_df.columns:
        numeric_aggs["duration"] = ["mean", "std", "min", "max"]

    for max_length in max_lengths:
        numeric_aggs[f"truncated_at_{max_length}"] = ["mean", "sum"]
        numeric_aggs[f"overflow_tokens_at_{max_length}"] = ["mean", "max"]

    grouped = (
        per_sample_df.groupby(["label"], dropna=False)
        .agg(numeric_aggs)
        .reset_index()
    )

    grouped.columns = [
        "__".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]

    grouped = grouped.rename(columns={"sample_id__count": "num_samples"})

    for max_length in max_lengths:
        rate_col = f"truncated_at_{max_length}__mean"
        sum_col = f"truncated_at_{max_length}__sum"
        if rate_col in grouped.columns:
            grouped = grouped.rename(columns={rate_col: f"truncation_rate_at_{max_length}"})
        if sum_col in grouped.columns:
            grouped = grouped.rename(columns={sum_col: f"num_truncated_at_{max_length}"})

    return grouped


def _make_summary_payload(
    per_sample_df: pd.DataFrame,
    by_label_df: pd.DataFrame,
    by_split_label_df: pd.DataFrame,
    max_lengths: list[int],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "num_samples": int(len(per_sample_df)),
        "splits": sorted(per_sample_df["split"].dropna().unique().tolist()),
        "labels": sorted(per_sample_df["label"].dropna().unique().tolist()),
        "max_lengths": [int(x) for x in max_lengths],
        "overall": {
            "token_len_mean": float(per_sample_df["token_len"].mean()),
            "token_len_std": float(per_sample_df["token_len"].std(ddof=0)),
            "token_len_min": int(per_sample_df["token_len"].min()),
            "token_len_max": int(per_sample_df["token_len"].max()),
            "word_count_mean": float(per_sample_df["word_count"].mean()),
            "word_count_std": float(per_sample_df["word_count"].std(ddof=0)),
        },
    }

    if "duration" in per_sample_df.columns:
        payload["overall"]["duration_mean"] = float(per_sample_df["duration"].mean())
        payload["overall"]["duration_std"] = float(per_sample_df["duration"].std(ddof=0))

    for max_length in max_lengths:
        col = f"truncated_at_{max_length}"
        overflow_col = f"overflow_tokens_at_{max_length}"
        payload["overall"][f"truncation_rate_at_{max_length}"] = float(per_sample_df[col].mean())
        payload["overall"][f"num_truncated_at_{max_length}"] = int(per_sample_df[col].sum())
        payload["overall"][f"max_overflow_tokens_at_{max_length}"] = int(per_sample_df[overflow_col].max())

    payload["by_label_preview"] = by_label_df.to_dict(orient="records")
    payload["by_split_label_preview"] = by_split_label_df.to_dict(orient="records")
    return payload


def main() -> int:
    args = build_parser().parse_args()

    max_lengths = sorted(set(int(x) for x in args.max_lengths))
    config = resolve_experiment_config(args.config)

    report_dir = ensure_dir(resolve_project_path(args.report_dir, start=PROJECT_ROOT))

    per_sample_df = _build_per_sample_report(
        config=config,
        max_lengths=max_lengths,
        use_normalized_text=bool(args.use_normalized_text),
    )
    by_split_label_df = _summarize_by_split_label(per_sample_df, max_lengths=max_lengths)
    by_label_df = _summarize_by_label(per_sample_df, max_lengths=max_lengths)

    write_csv(per_sample_df, report_dir / "per_sample_text_length.csv")
    write_csv(by_split_label_df, report_dir / "text_length_by_split_label.csv")
    write_csv(by_label_df, report_dir / "text_length_by_label.csv")

    payload = _make_summary_payload(
        per_sample_df=per_sample_df,
        by_label_df=by_label_df,
        by_split_label_df=by_split_label_df,
        max_lengths=max_lengths,
    )
    write_json(payload, report_dir / "text_length_summary.json")

    print(f"[OK] Wrote text length reports to: {report_dir}")
    print(json.dumps(payload["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())