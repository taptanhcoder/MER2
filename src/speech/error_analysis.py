from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_error_analysis_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    required_base = {"sample_id", "label_id", "pred_id", "confidence", "entropy", "top2_margin"}
    missing = [col for col in required_base if col not in pred_df.columns]
    if missing:
        raise ValueError(
            f"Prediction dataframe is missing required columns for error analysis: {missing}"
        )

    frame = pred_df.copy()

    label_col = "label" if "label" in frame.columns else "label_id"
    pred_col = "pred" if "pred" in frame.columns else "pred_id"

    frame["is_error"] = frame[label_col] != frame[pred_col]

    ordered_columns = [
        "sample_id",
        label_col,
        pred_col,
        "confidence",
        "entropy",
        "top2_margin",
        "audio_path",
        "group_id",
        "duration",
        "text",
        "raw_text",
        "split",
        "is_error",
        "probs",
        "logits",
    ]
    existing_columns = [col for col in ordered_columns if col in frame.columns]
    remaining_columns = [col for col in frame.columns if col not in existing_columns]
    return frame[existing_columns + remaining_columns]


def summarize_errors(error_df: pd.DataFrame) -> dict:
    summary: dict = {
        "num_rows": int(len(error_df)),
    }

    if "label" in error_df.columns:
        summary["label_counts"] = (
            error_df["label"].value_counts(dropna=False).sort_index().to_dict()
        )
    elif "label_id" in error_df.columns:
        summary["label_id_counts"] = (
            error_df["label_id"].value_counts(dropna=False).sort_index().to_dict()
        )

    if "pred" in error_df.columns:
        summary["pred_counts"] = (
            error_df["pred"].value_counts(dropna=False).sort_index().to_dict()
        )
    elif "pred_id" in error_df.columns:
        summary["pred_id_counts"] = (
            error_df["pred_id"].value_counts(dropna=False).sort_index().to_dict()
        )

    if "confidence" in error_df.columns and len(error_df) > 0:
        summary["confidence"] = {
            "mean": float(error_df["confidence"].mean()),
            "min": float(error_df["confidence"].min()),
            "max": float(error_df["confidence"].max()),
        }

    if "entropy" in error_df.columns and len(error_df) > 0:
        summary["entropy"] = {
            "mean": float(error_df["entropy"].mean()),
            "min": float(error_df["entropy"].min()),
            "max": float(error_df["entropy"].max()),
        }

    if "top2_margin" in error_df.columns and len(error_df) > 0:
        summary["top2_margin"] = {
            "mean": float(error_df["top2_margin"].mean()),
            "min": float(error_df["top2_margin"].min()),
            "max": float(error_df["top2_margin"].max()),
        }

    return summary


def save_basic_error_analysis(
    pred_df: pd.DataFrame,
    output_dir: str | Path,
    split_name: str,
    max_rows: int = 200,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame = build_error_analysis_frame(pred_df)
    error_frame = frame[frame["is_error"]].copy()

    if "confidence" in error_frame.columns:
        error_frame = error_frame.sort_values(
            by=["confidence", "top2_margin"],
            ascending=[False, True],
        )

    limited_error_frame = error_frame.head(int(max_rows)).copy()
    limited_error_frame.to_csv(output_path / f"{split_name}_errors.csv", index=False)

    summary = summarize_errors(error_frame)
    with (output_path / f"{split_name}_error_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)