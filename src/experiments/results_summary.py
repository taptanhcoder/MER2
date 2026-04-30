from __future__ import annotations

import re
from numbers import Number
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_yaml
from src.utils.io import ensure_dir, read_json, write_json, write_csv
from src.utils.paths import resolve_project_path


def resolve_experiment_name_from_config(config_path: str | Path) -> str:
    cfg = load_yaml(config_path)
    experiment = cfg.get("experiment", {})
    if isinstance(experiment, dict) and experiment.get("name"):
        return str(experiment["name"])
    return Path(config_path).stem


def expand_run_dirs_from_benchmark_config(
    benchmark_config_path: str | Path,
    project_root: str | Path | None = None,
) -> list[Path]:
    config_path = resolve_project_path(benchmark_config_path, start=project_root)
    cfg = load_yaml(config_path)
    benchmark_cfg = cfg.get("benchmark", {})

    experiments = benchmark_cfg.get("experiments", [])
    seeds = benchmark_cfg.get("seeds", [])
    output_root = resolve_project_path(
        benchmark_cfg.get("output_root", "outputs/runs"),
        start=project_root,
    )

    run_dirs: list[Path] = []
    for exp_config in experiments:
        exp_config_path = resolve_project_path(exp_config, start=project_root)
        exp_name = resolve_experiment_name_from_config(exp_config_path)
        for seed in seeds:
            run_dirs.append(output_root / exp_name / f"seed_{int(seed)}")
    return run_dirs


def _infer_experiment_name(run_dir: str | Path) -> str:
    return Path(run_dir).parent.name


def _infer_seed(run_dir: str | Path) -> int | None:
    match = re.search(r"seed_(\d+)", str(Path(run_dir).name))
    if match is None:
        return None
    return int(match.group(1))


def _is_scalar_number(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def flatten_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}

    fit_payload = payload.get("fit", {})
    if isinstance(fit_payload, dict):
        best_value = fit_payload.get("best_value")
        if _is_scalar_number(best_value):
            row["fit__best_value"] = float(best_value)

    for split in ("valid", "test"):
        split_payload = payload.get(split, {})
        if not isinstance(split_payload, dict):
            continue

        for key, value in split_payload.items():
            if key == "per_class" and isinstance(value, dict):
                for class_name, class_metrics in value.items():
                    if not isinstance(class_metrics, dict):
                        continue
                    for metric_name, metric_value in class_metrics.items():
                        if _is_scalar_number(metric_value):
                            row[
                                f"{split}__{class_name}__{metric_name}"
                            ] = float(metric_value)
            elif _is_scalar_number(value):
                row[f"{split}__{key}"] = float(value)

    return row


def build_runs_dataframe(run_dirs: list[str | Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        run_path = Path(run_dir)
        metrics_path = run_path / "metrics.json"
        if not metrics_path.exists():
            continue

        payload = read_json(metrics_path)
        row = {
            "run_dir": str(run_path),
            "experiment_name": _infer_experiment_name(run_path),
            "seed": _infer_seed(run_path),
        }
        row.update(flatten_metrics_payload(payload))
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["run_dir", "experiment_name", "seed"])

    df = pd.DataFrame(rows)
    if "seed" in df.columns:
        df = df.sort_values(["experiment_name", "seed"], kind="stable").reset_index(
            drop=True
        )
    return df


def aggregate_runs_dataframe(per_run_df: pd.DataFrame) -> pd.DataFrame:
    if per_run_df.empty:
        return pd.DataFrame(columns=["experiment_name", "num_runs"])

    numeric_cols = [
        col
        for col in per_run_df.columns
        if pd.api.types.is_numeric_dtype(per_run_df[col]) and col != "seed"
    ]

    rows: list[dict[str, Any]] = []
    grouped = per_run_df.groupby("experiment_name", sort=False)

    for experiment_name, group_df in grouped:
        row: dict[str, Any] = {
            "experiment_name": experiment_name,
            "num_runs": int(len(group_df)),
            "seeds": ",".join(str(int(seed)) for seed in group_df["seed"].dropna().tolist()),
        }

        for col in numeric_cols:
            values = group_df[col].dropna().astype(float)
            if values.empty:
                continue
            row[f"{col}__mean"] = float(values.mean())
            row[f"{col}__std"] = float(values.std(ddof=0))
            row[f"{col}__min"] = float(values.min())
            row[f"{col}__max"] = float(values.max())

        rows.append(row)

    return pd.DataFrame(rows)


def save_summary_reports(
    per_run_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    report_dir: str | Path,
) -> None:
    report_path = ensure_dir(report_dir)

    write_csv(per_run_df, report_path / "per_run_metrics.csv")
    write_csv(aggregate_df, report_path / "aggregate_metrics.csv")

    per_run_json = per_run_df.to_dict(orient="records")
    aggregate_json = aggregate_df.to_dict(orient="records")

    write_json(per_run_json, report_path / "per_run_metrics.json")
    write_json(aggregate_json, report_path / "aggregate_metrics.json")


def summarize_from_run_dirs(
    run_dirs: list[str | Path],
    report_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run_df = build_runs_dataframe(run_dirs)
    aggregate_df = aggregate_runs_dataframe(per_run_df)
    save_summary_reports(per_run_df, aggregate_df, report_dir)
    return per_run_df, aggregate_df