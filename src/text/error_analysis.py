from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_misclassification_frame(pred_df: pd.DataFrame) -> pd.DataFrame:
    if "label_id" not in pred_df.columns or "pred_id" not in pred_df.columns:
        raise ValueError("Prediction dataframe must contain label_id and pred_id columns")

    errors = pred_df[pred_df["label_id"] != pred_df["pred_id"]].copy()
    sort_cols = [col for col in ["confidence", "top2_margin"] if col in errors.columns]
    if sort_cols:
        errors = errors.sort_values(sort_cols, ascending=False)
    return errors.reset_index(drop=True)


def build_confusion_pairs_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    if "label" not in pred_df.columns or "pred" not in pred_df.columns:
        return pd.DataFrame(columns=["label", "pred", "count"])

    errors = pred_df[pred_df["label_id"] != pred_df["pred_id"]].copy()
    if errors.empty:
        return pd.DataFrame(columns=["label", "pred", "count"])

    summary = (
        errors.groupby(["label", "pred"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def save_basic_error_analysis(
    pred_df: pd.DataFrame,
    output_dir: str | Path,
    split_name: str = "test",
    max_rows: int = 200,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    errors = build_misclassification_frame(pred_df)
    confusion_pairs = build_confusion_pairs_summary(pred_df)

    errors.head(max_rows).to_csv(out_dir / f"{split_name}_errors.csv", index=False)
    confusion_pairs.to_csv(out_dir / f"{split_name}_confusion_pairs.csv", index=False)

    summary = {
        "split": split_name,
        "num_rows": int(len(pred_df)),
        "num_errors": int(len(errors)),
        "error_rate": float(len(errors) / len(pred_df)) if len(pred_df) > 0 else 0.0,
        "saved_top_errors": int(min(max_rows, len(errors))),
    }

    with (out_dir / f"{split_name}_error_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)