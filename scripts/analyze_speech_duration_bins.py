from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_BINS = [
    ("short", 0.0, 6.0),
    ("medium", 6.0, 12.0),
    ("long", 12.0, float("inf")),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze speech predictions by utterance duration bins."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help=(
            "Root directory containing experiment folders or seed folders. "
            "Example: outputs/runs/speech_duration_aware_tuning"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Prediction split to analyze.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="Directory where duration-bin reports will be written.",
    )
    parser.add_argument(
        "--label-names",
        type=str,
        nargs="*",
        default=["anger", "fear", "happiness", "sadness", "neutral"],
        help="Label names ordered by label_id.",
    )
    return parser


def _duration_bin(duration: float) -> str:
    for name, lo, hi in DEFAULT_BINS:
        if lo <= duration < hi:
            return name
    return "unknown"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _find_prediction_files(runs_root: Path, split: str) -> list[Path]:
    direct = sorted(runs_root.glob(f"seed_*/{split}_predictions.csv"))
    nested = sorted(runs_root.glob(f"*/seed_*/{split}_predictions.csv"))

    files = []
    seen = set()
    for path in direct + nested:
        resolved = str(path.resolve())
        if resolved not in seen:
            files.append(path)
            seen.add(resolved)

    return files


def _infer_experiment_and_seed(path: Path) -> tuple[str, int | None]:
    parts = path.parts

    seed = None
    for part in parts:
        if part.startswith("seed_"):
            try:
                seed = int(part.replace("seed_", ""))
            except ValueError:
                seed = None

    if path.parent.name.startswith("seed_"):
        experiment_name = path.parent.parent.name
    else:
        experiment_name = path.parent.name

    return experiment_name, seed


def _load_predictions(files: list[Path], label_names: list[str]) -> pd.DataFrame:
    rows = []

    for file_path in files:
        experiment_name, seed = _infer_experiment_and_seed(file_path)
        frame = pd.read_csv(file_path)

        required = ["label_id", "pred_id"]
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"{file_path} missing required columns: {missing}")

        if "duration" not in frame.columns:
            raise ValueError(
                f"{file_path} missing `duration` column. "
                "Speech prediction exports must include utterance duration for duration-bin analysis."
            )

        for _, row in frame.iterrows():
            label_id = _safe_int(row["label_id"])
            pred_id = _safe_int(row["pred_id"])
            duration = _safe_float(row["duration"])

            label = (
                label_names[label_id]
                if 0 <= label_id < len(label_names)
                else str(label_id)
            )
            pred = (
                label_names[pred_id]
                if 0 <= pred_id < len(label_names)
                else str(pred_id)
            )

            rows.append(
                {
                    "experiment_name": experiment_name,
                    "seed": seed,
                    "sample_id": str(row.get("sample_id", "")),
                    "label_id": label_id,
                    "pred_id": pred_id,
                    "label": label,
                    "pred": pred,
                    "duration": duration,
                    "duration_bin": _duration_bin(duration),
                    "confidence": _safe_float(row.get("confidence", 0.0)),
                    "entropy": _safe_float(row.get("entropy", 0.0)),
                    "top2_margin": _safe_float(row.get("top2_margin", 0.0)),
                    "correct": int(label_id == pred_id),
                }
            )

    if not rows:
        raise FileNotFoundError("No prediction rows were loaded.")

    return pd.DataFrame(rows)


def _classification_summary(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["label_id"].astype(int).tolist()
    y_pred = frame["pred_id"].astype(int).tolist()

    return {
        "num_samples": int(len(frame)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mean_confidence": float(frame["confidence"].mean()),
        "mean_entropy": float(frame["entropy"].mean()),
        "mean_top2_margin": float(frame["top2_margin"].mean()),
        "mean_duration": float(frame["duration"].mean()),
        "min_duration": float(frame["duration"].min()),
        "max_duration": float(frame["duration"].max()),
    }


def _summarize_by_group(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    records = []

    for group_values, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        record = {
            col: value for col, value in zip(group_cols, group_values)
        }
        record.update(_classification_summary(group))
        records.append(record)

    return pd.DataFrame(records).sort_values(group_cols).reset_index(drop=True)


def _summarize_seed_variability(frame: pd.DataFrame) -> pd.DataFrame:
    per_seed = _summarize_by_group(frame, ["experiment_name", "seed", "duration_bin"])

    records = []
    for (experiment_name, duration_bin), group in per_seed.groupby(
        ["experiment_name", "duration_bin"],
        dropna=False,
    ):
        values = group["macro_f1"].astype(float)
        records.append(
            {
                "experiment_name": experiment_name,
                "duration_bin": duration_bin,
                "num_seeds": int(group["seed"].nunique()),
                "macro_f1_mean": float(values.mean()),
                "macro_f1_std": float(values.std(ddof=0)),
                "macro_f1_min": float(values.min()),
                "macro_f1_max": float(values.max()),
                "macro_f1_range": float(values.max() - values.min()),
                "mean_confidence": float(group["mean_confidence"].mean()),
                "mean_duration": float(group["mean_duration"].mean()),
            }
        )

    return pd.DataFrame(records).sort_values(
        ["experiment_name", "duration_bin"]
    ).reset_index(drop=True)


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> int:
    args = build_parser().parse_args()

    runs_root = args.runs_root.resolve()
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    files = _find_prediction_files(runs_root, args.split)
    if not files:
        raise FileNotFoundError(
            f"No {args.split}_predictions.csv files found under {runs_root}"
        )

    predictions = _load_predictions(files, label_names=args.label_names)

    overall_by_experiment = _summarize_by_group(
        predictions,
        ["experiment_name"],
    )
    by_experiment_seed = _summarize_by_group(
        predictions,
        ["experiment_name", "seed"],
    )
    by_duration_bin = _summarize_by_group(
        predictions,
        ["experiment_name", "duration_bin"],
    )
    by_label_duration_bin = _summarize_by_group(
        predictions,
        ["experiment_name", "label", "duration_bin"],
    )
    seed_variability_by_bin = _summarize_seed_variability(predictions)

    _write_csv(predictions, report_dir / f"{args.split}_speech_predictions_with_duration_bins.csv")
    _write_csv(overall_by_experiment, report_dir / f"{args.split}_overall_by_experiment.csv")
    _write_csv(by_experiment_seed, report_dir / f"{args.split}_by_experiment_seed.csv")
    _write_csv(by_duration_bin, report_dir / f"{args.split}_by_duration_bin.csv")
    _write_csv(by_label_duration_bin, report_dir / f"{args.split}_by_label_duration_bin.csv")
    _write_csv(seed_variability_by_bin, report_dir / f"{args.split}_seed_variability_by_duration_bin.csv")

    summary = {
        "runs_root": str(runs_root),
        "split": args.split,
        "num_prediction_files": len(files),
        "num_rows": int(len(predictions)),
        "experiments": sorted(predictions["experiment_name"].dropna().unique().tolist()),
        "seeds": sorted(int(x) for x in predictions["seed"].dropna().unique().tolist()),
        "duration_bins": [
            {"name": name, "min_sec": lo, "max_sec": hi}
            for name, lo, hi in DEFAULT_BINS
        ],
        "overall_by_experiment": overall_by_experiment.to_dict(orient="records"),
    }
    _write_json(summary, report_dir / f"{args.split}_duration_bin_summary.json")

    print(f"[OK] Wrote speech duration-bin analysis to: {report_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())