from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_dir, write_csv, write_json
from src.utils.paths import resolve_project_path


GATE_COLUMNS = [
    "alpha_text_mean",
    "alpha_speech_mean",
    "alpha_interaction_mean",
]

BRANCH_PRED_COLUMNS = [
    "text_pred_id",
    "speech_pred_id",
    "interaction_pred_id",
]

BRANCH_CONF_COLUMNS = [
    "text_confidence",
    "speech_confidence",
    "interaction_confidence",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze fusion gate behavior from fusion prediction CSV files. "
            "This script is read-only and does not modify model outputs."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Fusion run directory containing valid_predictions.csv or test_predictions.csv. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help=(
            "Optional root directory containing experiment/seed_* fusion run directories. "
            "Example: outputs/runs/fusion_stability/fusion_light_bica_gate_v2"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["valid", "test"],
        help="Prediction split to analyze.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("outputs/reports/analysis/fusion_gates"),
        help="Directory where analysis CSV/JSON files will be written.",
    )
    return parser


def _load_label_names(run_dir: Path) -> list[str]:
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        return []

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    label_space = cfg.get("label_space", {})
    labels = label_space.get("labels", [])
    if isinstance(labels, list):
        return [str(label) for label in labels]

    return []


def _infer_seed(run_dir: Path) -> int | None:
    name = run_dir.name
    if not name.startswith("seed_"):
        return None

    try:
        return int(name.replace("seed_", ""))
    except ValueError:
        return None


def _infer_experiment_name(run_dir: Path) -> str:
    return run_dir.parent.name


def _prediction_path(run_dir: Path, split: str) -> Path:
    return run_dir / f"{split}_predictions.csv"


def _discover_run_dirs(run_dirs: list[Path], runs_root: Path | None) -> list[Path]:
    discovered: list[Path] = []

    for run_dir in run_dirs:
        resolved = resolve_project_path(run_dir, start=PROJECT_ROOT)
        discovered.append(resolved)

    if runs_root is not None:
        root = resolve_project_path(runs_root, start=PROJECT_ROOT)
        if root.exists():
            discovered.extend(sorted(path for path in root.glob("seed_*") if path.is_dir()))

    unique: list[Path] = []
    seen: set[str] = set()

    for path in discovered:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path.resolve())

    return unique


def _attach_label_names(frame: pd.DataFrame, label_names: list[str]) -> pd.DataFrame:
    result = frame.copy()

    if label_names:
        if "label_id" in result.columns and "label" not in result.columns:
            result["label"] = result["label_id"].map(
                lambda value: label_names[int(value)]
                if pd.notna(value) and int(value) < len(label_names)
                else str(value)
            )

        if "pred_id" in result.columns and "pred" not in result.columns:
            result["pred"] = result["pred_id"].map(
                lambda value: label_names[int(value)]
                if pd.notna(value) and int(value) < len(label_names)
                else str(value)
            )

    return result


def _load_prediction_frame(run_dir: Path, split: str) -> pd.DataFrame | None:
    path = _prediction_path(run_dir, split)
    if not path.exists():
        print(f"[WARN] Prediction file not found, skipped: {path}")
        return None

    frame = pd.read_csv(path)
    label_names = _load_label_names(run_dir)
    frame = _attach_label_names(frame, label_names)

    frame["run_dir"] = str(run_dir)
    frame["experiment_name"] = _infer_experiment_name(run_dir)
    frame["seed"] = _infer_seed(run_dir)
    frame["split"] = split

    if "label_id" in frame.columns and "pred_id" in frame.columns:
        frame["fusion_correct"] = frame["label_id"].astype(int) == frame["pred_id"].astype(int)

    return frame


def _available_columns(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in frame.columns]


def _mean_summary(
    frame: pd.DataFrame,
    group_cols: list[str],
    value_cols: list[str],
) -> pd.DataFrame:
    if frame.empty or not value_cols:
        return pd.DataFrame()

    summary = (
        frame.groupby(group_cols, dropna=False)[value_cols]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "__".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary


def _build_gate_by_class(frame: pd.DataFrame) -> pd.DataFrame:
    value_cols = _available_columns(frame, GATE_COLUMNS + BRANCH_CONF_COLUMNS)
    if "label" in frame.columns:
        return _mean_summary(frame, ["experiment_name", "seed", "split", "label"], value_cols)
    if "label_id" in frame.columns:
        return _mean_summary(frame, ["experiment_name", "seed", "split", "label_id"], value_cols)
    return pd.DataFrame()


def _build_gate_by_correctness(frame: pd.DataFrame) -> pd.DataFrame:
    if "fusion_correct" not in frame.columns:
        return pd.DataFrame()

    value_cols = _available_columns(frame, GATE_COLUMNS + BRANCH_CONF_COLUMNS)
    return _mean_summary(
        frame,
        ["experiment_name", "seed", "split", "fusion_correct"],
        value_cols,
    )


def _build_branch_agreement(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["text_pred_id", "speech_pred_id", "label_id", "pred_id"]
    if any(column not in frame.columns for column in required):
        return pd.DataFrame()

    result = frame.copy()
    result["text_speech_agree"] = (
        result["text_pred_id"].astype(int) == result["speech_pred_id"].astype(int)
    )
    result["fusion_matches_text"] = (
        result["pred_id"].astype(int) == result["text_pred_id"].astype(int)
    )
    result["fusion_matches_speech"] = (
        result["pred_id"].astype(int) == result["speech_pred_id"].astype(int)
    )

    if "interaction_pred_id" in result.columns:
        result["fusion_matches_interaction"] = (
            result["pred_id"].astype(int) == result["interaction_pred_id"].astype(int)
        )

    bool_cols = [
        column
        for column in [
            "text_speech_agree",
            "fusion_matches_text",
            "fusion_matches_speech",
            "fusion_matches_interaction",
            "fusion_correct",
        ]
        if column in result.columns
    ]

    summary = (
        result.groupby(["experiment_name", "seed", "split"], dropna=False)[bool_cols]
        .mean()
        .reset_index()
    )
    return summary


def _build_error_cases(frame: pd.DataFrame) -> pd.DataFrame:
    if "fusion_correct" not in frame.columns:
        return pd.DataFrame()

    errors = frame[~frame["fusion_correct"]].copy()
    if errors.empty:
        return errors

    preferred_cols = [
        "experiment_name",
        "seed",
        "split",
        "sample_id",
        "label",
        "pred",
        "label_id",
        "pred_id",
        "confidence",
        "entropy",
        "top2_margin",
        "alpha_text_mean",
        "alpha_speech_mean",
        "alpha_interaction_mean",
        "text_pred_id",
        "speech_pred_id",
        "interaction_pred_id",
        "text_confidence",
        "speech_confidence",
        "interaction_confidence",
        "text",
        "raw_text",
        "audio_path",
        "group_id",
    ]

    available = [column for column in preferred_cols if column in errors.columns]
    return errors[available]


def _build_overall_summary(frame: pd.DataFrame) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "num_rows": int(len(frame)),
        "experiments": sorted(frame["experiment_name"].dropna().unique().tolist())
        if "experiment_name" in frame.columns
        else [],
        "seeds": sorted(int(seed) for seed in frame["seed"].dropna().unique().tolist())
        if "seed" in frame.columns
        else [],
        "available_gate_columns": _available_columns(frame, GATE_COLUMNS),
        "available_branch_prediction_columns": _available_columns(frame, BRANCH_PRED_COLUMNS),
        "available_branch_confidence_columns": _available_columns(frame, BRANCH_CONF_COLUMNS),
    }

    if "fusion_correct" in frame.columns:
        payload["fusion_accuracy_from_predictions"] = float(frame["fusion_correct"].mean())

    for column in _available_columns(frame, GATE_COLUMNS + BRANCH_CONF_COLUMNS):
        payload[f"{column}_mean"] = float(frame[column].mean())
        payload[f"{column}_std"] = float(frame[column].std(ddof=0))

    return payload


def main() -> int:
    args = build_parser().parse_args()

    run_dirs = _discover_run_dirs(args.run_dir, args.runs_root)
    if not run_dirs:
        raise ValueError("No run directories were provided or discovered.")

    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        frame = _load_prediction_frame(run_dir, split=args.split)
        if frame is not None:
            frames.append(frame)

    if not frames:
        raise FileNotFoundError("No prediction CSV files were loaded.")

    all_predictions = pd.concat(frames, ignore_index=True)
    report_dir = ensure_dir(resolve_project_path(args.report_dir, start=PROJECT_ROOT))

    gate_by_class = _build_gate_by_class(all_predictions)
    gate_by_correctness = _build_gate_by_correctness(all_predictions)
    branch_agreement = _build_branch_agreement(all_predictions)
    error_cases = _build_error_cases(all_predictions)
    overall_summary = _build_overall_summary(all_predictions)

    write_csv(all_predictions, report_dir / f"{args.split}_fusion_predictions_merged.csv")

    if not gate_by_class.empty:
        write_csv(gate_by_class, report_dir / f"{args.split}_gate_by_class.csv")

    if not gate_by_correctness.empty:
        write_csv(gate_by_correctness, report_dir / f"{args.split}_gate_by_correctness.csv")

    if not branch_agreement.empty:
        write_csv(branch_agreement, report_dir / f"{args.split}_branch_agreement.csv")

    if not error_cases.empty:
        write_csv(error_cases, report_dir / f"{args.split}_fusion_error_cases.csv")

    write_json(overall_summary, report_dir / f"{args.split}_gate_summary.json")

    print(f"[OK] Wrote fusion gate analysis to: {report_dir}")
    print(json.dumps(overall_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())