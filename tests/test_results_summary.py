from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.results_summary import (
    aggregate_runs_dataframe,
    build_runs_dataframe,
)
from src.utils.io import write_json


def _write_metrics(run_dir: Path, valid_macro_f1: float, test_macro_f1: float) -> None:
    payload = {
        "fit": {"best_value": valid_macro_f1},
        "valid": {
            "macro_f1": valid_macro_f1,
            "uar": valid_macro_f1,
            "per_class": {
                "anger": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 6}
            },
        },
        "test": {
            "macro_f1": test_macro_f1,
            "uar": test_macro_f1,
            "per_class": {
                "anger": {"precision": 0.6, "recall": 0.6, "f1": 0.6, "support": 6}
            },
        },
    }
    write_json(payload, run_dir / "metrics.json")


def test_results_summary_builds_and_aggregates(tmp_path: Path) -> None:
    run_a = tmp_path / "speech_hubert_base_attn_freeze_weighted_ce" / "seed_42"
    run_b = tmp_path / "speech_hubert_base_attn_freeze_weighted_ce" / "seed_52"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    _write_metrics(run_a, valid_macro_f1=0.40, test_macro_f1=0.60)
    _write_metrics(run_b, valid_macro_f1=0.50, test_macro_f1=0.70)

    per_run_df = build_runs_dataframe([run_a, run_b])
    aggregate_df = aggregate_runs_dataframe(per_run_df)

    assert not per_run_df.empty
    assert len(per_run_df) == 2
    assert "valid__macro_f1" in per_run_df.columns
    assert "test__anger__f1" in per_run_df.columns

    assert not aggregate_df.empty
    row = aggregate_df.iloc[0]
    assert row["experiment_name"] == "speech_hubert_base_attn_freeze_weighted_ce"
    assert row["num_runs"] == 2
    assert abs(row["valid__macro_f1__mean"] - 0.45) < 1e-8
    assert abs(row["test__macro_f1__mean"] - 0.65) < 1e-8