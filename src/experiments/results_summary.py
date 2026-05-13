from __future__ import annotations

import re
from numbers import Number
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import load_yaml
from src.utils.io import ensure_dir, read_json, write_csv, write_json
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


def _history_metric(
    history_item: dict[str, Any],
    split: str,
    metric_name: str,
) -> float | None:
    split_payload = history_item.get(split)
    if not isinstance(split_payload, dict):
        return None

    value = split_payload.get(metric_name)
    if not _is_scalar_number(value):
        return None

    return float(value)


def _add_fit_history_diagnostics(row: dict[str, Any], fit_payload: dict[str, Any]) -> None:
    history = fit_payload.get("history", [])
    if not isinstance(history, list) or not history:
        return

    valid_candidates: list[dict[str, float | int]] = []
    train_values: list[float] = []

    for idx, item in enumerate(history, start=1):
        if not isinstance(item, dict):
            continue

        epoch = int(item.get("epoch", idx))
        train_macro_f1 = _history_metric(item, "train", "macro_f1")
        valid_macro_f1 = _history_metric(item, "valid", "macro_f1")

        if train_macro_f1 is not None:
            train_values.append(train_macro_f1)

        if valid_macro_f1 is not None:
            candidate: dict[str, float | int] = {
                "epoch": epoch,
                "valid_macro_f1": valid_macro_f1,
            }
            if train_macro_f1 is not None:
                candidate["train_macro_f1"] = train_macro_f1
                candidate["train_valid_gap"] = train_macro_f1 - valid_macro_f1
            valid_candidates.append(candidate)

    row["fit__num_epochs_run"] = float(len(history))

    if train_values:
        row["fit__best_train_macro_f1"] = max(train_values)
        row["fit__last_train_macro_f1"] = train_values[-1]

    if valid_candidates:
        best_valid = max(valid_candidates, key=lambda x: float(x["valid_macro_f1"]))

        row["fit__best_valid_epoch"] = float(best_valid["epoch"])
        row["fit__best_valid_macro_f1"] = float(best_valid["valid_macro_f1"])

        if "train_macro_f1" in best_valid:
            row["fit__train_macro_f1_at_best_valid"] = float(best_valid["train_macro_f1"])

        if "train_valid_gap" in best_valid:
            row["fit__train_valid_gap_at_best_valid"] = float(best_valid["train_valid_gap"])

        last_valid = valid_candidates[-1]
        row["fit__last_valid_macro_f1"] = float(last_valid["valid_macro_f1"])

        if "train_valid_gap" in last_valid:
            row["fit__last_train_valid_gap"] = float(last_valid["train_valid_gap"])


def flatten_metrics_payload(payload: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}

    fit_payload = payload.get("fit", {})
    if isinstance(fit_payload, dict):
        best_value = fit_payload.get("best_value")
        if _is_scalar_number(best_value):
            row["fit__best_value"] = float(best_value)

        _add_fit_history_diagnostics(row, fit_payload)

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
                            row[f"{split}__{class_name}__{metric_name}"] = float(metric_value)

            elif _is_scalar_number(value):
                row[f"{split}__{key}"] = float(value)

    valid_macro_f1 = row.get("valid__macro_f1")
    test_macro_f1 = row.get("test__macro_f1")
    if _is_scalar_number(valid_macro_f1) and _is_scalar_number(test_macro_f1):
        row["diagnostic__valid_test_macro_f1_gap"] = float(valid_macro_f1) - float(test_macro_f1)

    train_valid_gap = row.get("fit__train_valid_gap_at_best_valid")
    if _is_scalar_number(train_valid_gap):
        row["diagnostic__overfit_gap"] = float(train_valid_gap)

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
            "seeds": ",".join(
                str(int(seed)) for seed in group_df["seed"].dropna().tolist()
            ),
        }

        for col in numeric_cols:
            values = group_df[col].dropna().astype(float)
            if values.empty:
                continue

            mean_value = float(values.mean())
            std_value = float(values.std(ddof=0))
            min_value = float(values.min())
            max_value = float(values.max())

            row[f"{col}__mean"] = mean_value
            row[f"{col}__std"] = std_value
            row[f"{col}__min"] = min_value
            row[f"{col}__max"] = max_value
            row[f"{col}__range"] = max_value - min_value
            row[f"{col}__mean_minus_std"] = mean_value - std_value

        rows.append(row)

    return pd.DataFrame(rows)


def _aggregate_value(row: pd.Series, key: str) -> float | None:
    value = row.get(key)
    if _is_scalar_number(value):
        return float(value)
    return None


def build_stability_summary(aggregate_df: pd.DataFrame) -> pd.DataFrame:
    if aggregate_df.empty:
        return pd.DataFrame(
            columns=[
                "experiment_name",
                "num_runs",
                "seeds",
                "test_macro_f1_mean",
                "test_macro_f1_std",
                "test_macro_f1_range",
                "valid_macro_f1_mean",
                "valid_macro_f1_std",
                "overfit_gap_mean",
                "valid_test_gap_mean",
                "stability_score",
            ]
        )

    rows: list[dict[str, Any]] = []

    for _, source_row in aggregate_df.iterrows():
        test_mean = _aggregate_value(source_row, "test__macro_f1__mean")
        test_std = _aggregate_value(source_row, "test__macro_f1__std")
        test_range = _aggregate_value(source_row, "test__macro_f1__range")

        valid_mean = _aggregate_value(source_row, "valid__macro_f1__mean")
        valid_std = _aggregate_value(source_row, "valid__macro_f1__std")

        overfit_gap_mean = _aggregate_value(
            source_row,
            "diagnostic__overfit_gap__mean",
        )
        valid_test_gap_mean = _aggregate_value(
            source_row,
            "diagnostic__valid_test_macro_f1_gap__mean",
        )

        stability_score = None
        if test_mean is not None and test_std is not None:
            stability_score = test_mean - test_std

        rows.append(
            {
                "experiment_name": source_row["experiment_name"],
                "num_runs": int(source_row["num_runs"]),
                "seeds": source_row.get("seeds", ""),
                "test_macro_f1_mean": test_mean,
                "test_macro_f1_std": test_std,
                "test_macro_f1_min": _aggregate_value(source_row, "test__macro_f1__min"),
                "test_macro_f1_max": _aggregate_value(source_row, "test__macro_f1__max"),
                "test_macro_f1_range": test_range,
                "valid_macro_f1_mean": valid_mean,
                "valid_macro_f1_std": valid_std,
                "overfit_gap_mean": overfit_gap_mean,
                "valid_test_gap_mean": valid_test_gap_mean,
                "stability_score": stability_score,
            }
        )

    frame = pd.DataFrame(rows)
    if "stability_score" in frame.columns:
        frame = frame.sort_values(
            ["stability_score", "test_macro_f1_mean"],
            ascending=[False, False],
            kind="stable",
        ).reset_index(drop=True)

    return frame


def save_summary_reports(
    per_run_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    report_dir: str | Path,
) -> None:
    report_path = ensure_dir(report_dir)

    stability_df = build_stability_summary(aggregate_df)

    write_csv(per_run_df, report_path / "per_run_metrics.csv")
    write_csv(aggregate_df, report_path / "aggregate_metrics.csv")
    write_csv(stability_df, report_path / "stability_summary.csv")

    write_json(
        per_run_df.to_dict(orient="records"),
        report_path / "per_run_metrics.json",
    )
    write_json(
        aggregate_df.to_dict(orient="records"),
        report_path / "aggregate_metrics.json",
    )
    write_json(
        stability_df.to_dict(orient="records"),
        report_path / "stability_summary.json",
    )


def summarize_from_run_dirs(
    run_dirs: list[str | Path],
    report_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run_df = build_runs_dataframe(run_dirs)
    aggregate_df = aggregate_runs_dataframe(per_run_df)
    save_summary_reports(per_run_df, aggregate_df, report_dir)
    return per_run_df, aggregate_df