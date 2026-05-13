from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import resolve_project_path


FOLD_RE = re.compile(r"fold_(\d+)")
SEED_RE = re.compile(r"seed_(\d+)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize K-fold CV results from metrics.json or prediction CSVs. "
            "Supports fold/seed parsing from paths such as "
            "`..._fold_00_seed_42/seed_42/metrics.json`."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help=(
            "Root containing fold/seed runs. The script searches recursively for "
            "metrics.json and prediction CSV files."
        ),
    )
    parser.add_argument(
        "--prediction-pattern",
        type=str,
        default="test_predictions.csv",
        help="Prediction filename to search for when metrics.json is absent.",
    )
    parser.add_argument(
        "--metrics-pattern",
        type=str,
        default="metrics.json",
        help="Metrics filename to search for.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output report directory.",
    )
    parser.add_argument(
        "--include-substring",
        type=str,
        default=None,
        help=(
            "Only include files whose full path contains this substring. "
            "Use `fold_` to include only CV fold-specific runs."
        ),
    )
    parser.add_argument(
        "--exclude-substring",
        action="append",
        default=[],
        help=(
            "Exclude files whose full path contains this substring. "
            "Can be provided multiple times."
        ),
    )
    parser.add_argument(
        "--require-fold",
        action="store_true",
        help="Drop rows where fold cannot be parsed from the path.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Override inferred model name for all collected rows. Useful for a "
            "single CV experiment such as final_light_bica_gate."
        ),
    )
    return parser


def _resolve(path: str | Path) -> Path:
    return resolve_project_path(path, start=PROJECT_ROOT)


def _path_text(path: Path) -> str:
    return str(path).replace("\\", "/")


def _path_allowed(
    path: Path,
    include_substring: str | None,
    exclude_substrings: list[str],
) -> bool:
    text = _path_text(path)

    if include_substring and include_substring not in text:
        return False

    for item in exclude_substrings:
        if item and item in text:
            return False

    return True


def _parse_fold(path: Path) -> str:
    text = _path_text(path)
    match = FOLD_RE.search(text)
    if not match:
        return "unknown"
    return f"fold_{int(match.group(1)):02d}"


def _parse_seed(path: Path) -> str:
    text = _path_text(path)
    matches = SEED_RE.findall(text)
    if not matches:
        return "unknown"
    return str(int(matches[-1]))


def _strip_fold_seed_suffix(name: str) -> str:
    # Example:
    # fusion_light_bica_gate_xxx_fold_00_seed_42
    # -> fusion_light_bica_gate_xxx
    return re.sub(r"_fold_\d+_seed_\d+$", "", name)


def _infer_model(path: Path, explicit_model_name: str | None = None) -> str:
    if explicit_model_name:
        return explicit_model_name

    parts = list(path.parts)

    # Prefer the path component that explicitly carries fold/seed suffix.
    for part in reversed(parts):
        if FOLD_RE.search(part) and SEED_RE.search(part):
            cleaned = _strip_fold_seed_suffix(part)
            return cleaned or part

    # If path structure is .../<model>/fold_00/seed_42/metrics.json,
    # use the component immediately before fold_XX.
    for idx, part in enumerate(parts):
        if FOLD_RE.fullmatch(part) and idx > 0:
            return parts[idx - 1]

    # Fallback: use parent directory.
    if path.parent.name.startswith("seed_") and path.parent.parent.name:
        return _strip_fold_seed_suffix(path.parent.parent.name)

    return _strip_fold_seed_suffix(path.parent.name) or "cv_model"


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return None
        return float(value)
    return None


def _metrics_from_predictions(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)

    required = ["label_id", "pred_id"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    y_true = frame["label_id"].astype(int).to_numpy()
    y_pred = frame["pred_id"].astype(int).to_numpy()

    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "test_macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "test_macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "test_uar": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def _flatten_per_class_metrics(
    data: dict[str, Any],
    prefix: str = "test",
) -> dict[str, float]:
    row: dict[str, float] = {}

    per_class = data.get("per_class")
    if not isinstance(per_class, dict):
        return row

    for label, metrics in per_class.items():
        if not isinstance(metrics, dict):
            continue
        safe_label = str(label).replace(" ", "_").replace("/", "_")
        for metric_name, value in metrics.items():
            numeric = _safe_float(value)
            if numeric is None:
                continue
            row[f"{prefix}_per_class_{safe_label}_{metric_name}"] = numeric

    return row


def _flatten_metrics_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    row: dict[str, Any] = {}

    # Format A:
    # {"valid": {"macro_f1": ...}, "test": {"macro_f1": ...}}
    has_split_blocks = False
    for split in ["train", "valid", "test"]:
        split_data = data.get(split)
        if isinstance(split_data, dict):
            has_split_blocks = True
            for key, value in split_data.items():
                numeric = _safe_float(value)
                if numeric is not None:
                    row[f"{split}_{key}"] = numeric

            # Optional split-level per-class metrics.
            row.update(_flatten_per_class_metrics(split_data, prefix=split))

    # Format B:
    # {"accuracy": ..., "macro_f1": ..., "per_class": {...}}
    # Treat flat metrics as test metrics, which matches run_fusion.py output.
    if not has_split_blocks:
        for key, value in data.items():
            numeric = _safe_float(value)
            if numeric is not None:
                row[f"test_{key}"] = numeric

        row.update(_flatten_per_class_metrics(data, prefix="test"))

    return row


def _collect_metric_paths(
    runs_root: Path,
    metrics_pattern: str,
    include_substring: str | None,
    exclude_substrings: list[str],
) -> list[Path]:
    paths = sorted(runs_root.rglob(metrics_pattern))
    return [
        path
        for path in paths
        if _path_allowed(path, include_substring, exclude_substrings)
    ]


def _collect_prediction_paths(
    runs_root: Path,
    prediction_pattern: str,
    include_substring: str | None,
    exclude_substrings: list[str],
) -> list[Path]:
    paths = sorted(runs_root.rglob(prediction_pattern))
    return [
        path
        for path in paths
        if _path_allowed(path, include_substring, exclude_substrings)
    ]


def _base_row(
    path: Path,
    explicit_model_name: str | None,
) -> dict[str, Any]:
    return {
        "source": _path_text(path),
        "model": _infer_model(path, explicit_model_name),
        "fold": _parse_fold(path),
        "seed": _parse_seed(path),
    }


def _collect_rows(
    runs_root: Path,
    metrics_pattern: str,
    prediction_pattern: str,
    include_substring: str | None,
    exclude_substrings: list[str],
    explicit_model_name: str | None,
    require_fold: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    metric_paths = _collect_metric_paths(
        runs_root=runs_root,
        metrics_pattern=metrics_pattern,
        include_substring=include_substring,
        exclude_substrings=exclude_substrings,
    )

    metric_dirs = set()
    for path in metric_paths:
        row = _base_row(path, explicit_model_name)
        if require_fold and row["fold"] == "unknown":
            continue

        row.update(_flatten_metrics_json(path))
        rows.append(row)
        metric_dirs.add(path.parent.resolve())

    prediction_paths = _collect_prediction_paths(
        runs_root=runs_root,
        prediction_pattern=prediction_pattern,
        include_substring=include_substring,
        exclude_substrings=exclude_substrings,
    )

    # Add prediction-based rows only when a metrics file is not already present
    # in the same directory.
    for path in prediction_paths:
        if path.parent.resolve() in metric_dirs:
            continue

        row = _base_row(path, explicit_model_name)
        if require_fold and row["fold"] == "unknown":
            continue

        row.update(_metrics_from_predictions(path))
        rows.append(row)

    return rows


def _ci95(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")

    if len(values) == 1:
        value = float(values[0])
        return value, value

    mean = float(values.mean())
    std = float(values.std(ddof=1))
    n = int(len(values))

    half_width = 1.96 * std / math.sqrt(n)
    return mean - half_width, mean + half_width


def _fold_sort_key(value: str) -> tuple[int, str]:
    match = FOLD_RE.fullmatch(str(value))
    if match:
        return int(match.group(1)), str(value)
    return 10_000, str(value)


def _seed_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), str(value)
    except Exception:
        return 10_000, str(value)


def _aggregate(per_run: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col for col in per_run.columns
        if col.startswith("test_") and pd.api.types.is_numeric_dtype(per_run[col])
    ]

    records = []

    for model_name, group in per_run.groupby("model", dropna=False):
        folds = sorted(group["fold"].astype(str).unique().tolist(), key=_fold_sort_key)
        seeds = sorted(group["seed"].astype(str).unique().tolist(), key=_seed_sort_key)

        record: dict[str, Any] = {
            "model": model_name,
            "num_runs": int(len(group)),
            "folds": ",".join(folds),
            "seeds": ",".join(seeds),
        }

        for col in metric_cols:
            values = group[col].dropna().astype(float).to_numpy()
            if len(values) == 0:
                continue

            ci_low, ci_high = _ci95(values)
            record[f"{col}_mean"] = float(values.mean())
            record[f"{col}_std"] = float(values.std(ddof=0))
            record[f"{col}_min"] = float(values.min())
            record[f"{col}_max"] = float(values.max())
            record[f"{col}_range"] = float(values.max() - values.min())
            record[f"{col}_ci95_low"] = float(ci_low)
            record[f"{col}_ci95_high"] = float(ci_high)

        records.append(record)

    return pd.DataFrame(records)


def _build_fold_table(per_run: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col for col in per_run.columns
        if col.startswith("test_") and pd.api.types.is_numeric_dtype(per_run[col])
    ]

    keep_cols = ["model", "fold", "seed", *metric_cols, "source"]
    keep_cols = [col for col in keep_cols if col in per_run.columns]

    table = per_run[keep_cols].copy()
    table = table.sort_values(
        by=["model", "fold", "seed"],
        key=lambda series: series.map(
            _fold_sort_key if series.name == "fold" else (
                _seed_sort_key if series.name == "seed" else lambda x: (0, str(x))
            )
        ),
    )

    return table


def main() -> int:
    args = build_parser().parse_args()

    runs_root = _resolve(args.runs_root)
    output_dir = _resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    rows = _collect_rows(
        runs_root=runs_root,
        metrics_pattern=str(args.metrics_pattern),
        prediction_pattern=str(args.prediction_pattern),
        include_substring=args.include_substring,
        exclude_substrings=list(args.exclude_substring or []),
        explicit_model_name=args.model_name,
        require_fold=bool(args.require_fold),
    )

    if not rows:
        raise RuntimeError(
            f"No matching {args.metrics_pattern} or {args.prediction_pattern} files "
            f"found under {runs_root}. include_substring={args.include_substring!r}"
        )

    per_run = pd.DataFrame(rows)

    # Stable ordering for inspection.
    per_run = per_run.sort_values(
        by=["model", "fold", "seed", "source"],
        key=lambda series: series.map(
            _fold_sort_key if series.name == "fold" else (
                _seed_sort_key if series.name == "seed" else lambda x: (0, str(x))
            )
        ),
    ).reset_index(drop=True)

    aggregate = _aggregate(per_run)
    fold_table = _build_fold_table(per_run)

    per_run.to_csv(output_dir / "cv_per_run_metrics.csv", index=False)
    aggregate.to_csv(output_dir / "cv_aggregate_metrics.csv", index=False)
    fold_table.to_csv(output_dir / "cv_fold_metrics.csv", index=False)

    (output_dir / "cv_per_run_metrics.json").write_text(
        json.dumps(per_run.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "cv_aggregate_metrics.json").write_text(
        json.dumps(aggregate.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "cv_fold_metrics.json").write_text(
        json.dumps(fold_table.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Wrote CV summary to: {output_dir}")
    print(json.dumps(aggregate.to_dict(orient="records"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())