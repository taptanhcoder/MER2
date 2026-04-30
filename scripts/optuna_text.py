from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Optuna is required for this script. Install it with: pip install optuna"
    ) from exc

from src.utils.config import load_yaml
from src.utils.io import write_json, write_yaml
from src.utils.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Optuna hyperparameter search for the text branch without modifying the stable baseline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the Optuna search config YAML.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "stage_a", "stage_b"],
        help="Run only stage_a, only stage_b, or both.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device passed through to train_text.py (cpu, cuda, auto).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous stage output directories before running.",
    )
    return parser


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _set_nested(container: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = container
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _suggest_param(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    kind = str(spec["type"]).lower()
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if kind == "float":
        step = spec.get("step")
        log = bool(spec.get("log", False))
        if step is not None:
            return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), step=float(step), log=log)
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=log)
    if kind == "int":
        step = int(spec.get("step", 1))
        log = bool(spec.get("log", False))
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]), step=step, log=log)
    raise KeyError(f"Unsupported search-space type: {kind}")


def _build_trial_overrides(trial: optuna.Trial, space_cfg: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for dotted_key, spec in space_cfg.items():
        value = _suggest_param(trial, dotted_key, dict(spec))
        _set_nested(overrides, dotted_key, value)
    return overrides


def _load_base_experiment(base_experiment_path: Path) -> dict[str, Any]:
    return load_yaml(base_experiment_path)


def _materialize_experiment_config(
    base_experiment_path: Path,
    overrides: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    base_cfg = _load_base_experiment(base_experiment_path)
    merged = _deep_update(base_cfg, overrides)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(merged, output_path)
    return merged


def _run_train_text(
    experiment_config_path: Path,
    output_root: Path,
    seed: int,
    device: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/train_text.py",
        "--config",
        str(experiment_config_path),
        "--device",
        device,
        "--seed",
        str(seed),
        "--output-root",
        str(output_root),
    ]
    process = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "train_text.py failed.\n"
            f"STDOUT:\n{process.stdout}\n\n"
            f"STDERR:\n{process.stderr}"
        )

    experiment_cfg = load_yaml(experiment_config_path)
    experiment_name = str(experiment_cfg["experiment"]["name"])
    run_dir = output_root / experiment_name / f"seed_{seed}"
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found after training: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics_payload = json.load(f)

    return {
        "run_dir": str(run_dir.resolve()),
        "metrics": metrics_payload,
        "stdout": process.stdout,
        "stderr": process.stderr,
    }


def _extract_valid_macro_f1(metrics_payload: dict[str, Any]) -> float:
    valid_section = metrics_payload.get("valid")
    if not isinstance(valid_section, dict):
        raise KeyError("`valid` section not found in metrics.json")
    if "macro_f1" not in valid_section:
        raise KeyError("`valid.macro_f1` not found in metrics.json")
    return float(valid_section["macro_f1"])


def _extract_test_macro_f1(metrics_payload: dict[str, Any]) -> float:
    test_section = metrics_payload.get("test")
    if not isinstance(test_section, dict):
        raise KeyError("`test` section not found in metrics.json")
    if "macro_f1" not in test_section:
        raise KeyError("`test.macro_f1` not found in metrics.json")
    return float(test_section["macro_f1"])


def _prepare_sampler(study_cfg: dict[str, Any]) -> optuna.samplers.BaseSampler:
    sampler_cfg = dict(study_cfg.get("sampler", {}))
    sampler_name = str(sampler_cfg.get("name", "tpe")).lower()
    if sampler_name != "tpe":
        raise KeyError(f"Unsupported sampler: {sampler_name}")
    return TPESampler(
        seed=int(sampler_cfg.get("seed", 2026)),
        multivariate=bool(sampler_cfg.get("multivariate", True)),
    )


def _prepare_pruner(study_cfg: dict[str, Any]) -> optuna.pruners.BasePruner:
    pruner_cfg = dict(study_cfg.get("pruner", {}))
    pruner_name = str(pruner_cfg.get("name", "median")).lower()
    if pruner_name != "median":
        raise KeyError(f"Unsupported pruner: {pruner_name}")
    return MedianPruner(
        n_startup_trials=int(pruner_cfg.get("n_startup_trials", 8)),
        n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 4)),
        interval_steps=int(pruner_cfg.get("interval_steps", 1)),
    )


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def run_stage_a(
    search_cfg: dict[str, Any],
    study_cfg: dict[str, Any],
    space_cfg: dict[str, Any],
    device: str,
    overwrite: bool,
) -> dict[str, Any]:
    base_experiment_path = resolve_project_path(search_cfg["base_experiment"], start=PROJECT_ROOT)
    stage_a_cfg = dict(search_cfg["stage_a"])
    output_root = resolve_project_path(stage_a_cfg["output_root"], start=PROJECT_ROOT)
    stage_dir = output_root / "artifacts"
    trials_dir = stage_dir / "trial_configs"

    if overwrite:
        _safe_rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)

    sampler = _prepare_sampler(study_cfg)
    pruner = _prepare_pruner(study_cfg)

    storage = study_cfg.get("storage")
    study = optuna.create_study(
        study_name=str(study_cfg["name"]) + "_stage_a",
        direction=str(study_cfg.get("direction", "maximize")),
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=bool(storage),
    )

    objective_metric = str(search_cfg.get("objective_metric", "valid_macro_f1"))
    if objective_metric != "valid_macro_f1":
        raise KeyError(f"Unsupported objective_metric: {objective_metric}")

    seed = int(stage_a_cfg["seed"])
    trial_records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        overrides = _build_trial_overrides(trial, space_cfg)
        experiment_name = f"{study_cfg['name']}_trial_{trial.number:03d}"
        overrides = _deep_update(
            overrides,
            {"experiment": {"name": experiment_name}},
        )

        trial_cfg_path = trials_dir / f"trial_{trial.number:03d}.yaml"
        materialized_cfg = _materialize_experiment_config(
            base_experiment_path=base_experiment_path,
            overrides=overrides,
            output_path=trial_cfg_path,
        )

        run_payload = _run_train_text(
            experiment_config_path=trial_cfg_path,
            output_root=output_root,
            seed=seed,
            device=device,
        )
        valid_macro_f1 = _extract_valid_macro_f1(run_payload["metrics"])
        test_macro_f1 = _extract_test_macro_f1(run_payload["metrics"])

        record = {
            "trial_number": trial.number,
            "seed": seed,
            "experiment_name": materialized_cfg["experiment"]["name"],
            "config_path": str(trial_cfg_path.resolve()),
            "run_dir": run_payload["run_dir"],
            "valid_macro_f1": valid_macro_f1,
            "test_macro_f1": test_macro_f1,
            "params": trial.params,
        }
        trial_records.append(record)
        write_json(
            {
                "stage": "stage_a",
                "records": trial_records,
            },
            output_root / "stage_a_records.json",
        )
        return valid_macro_f1

    study.optimize(objective, n_trials=int(stage_a_cfg["n_trials"]))

    completed_trials = [
        {
            "trial_number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        }
        for t in study.trials
    ]
    completed_trials.sort(
        key=lambda item: float(item["value"]) if item["value"] is not None else -math.inf,
        reverse=True,
    )

    stage_a_summary = {
        "study_name": str(study_cfg["name"]) + "_stage_a",
        "best_trial_number": study.best_trial.number if study.best_trial is not None else None,
        "best_value": float(study.best_value) if study.best_trial is not None else None,
        "best_params": study.best_params if study.best_trial is not None else {},
        "top_k_for_stage_b": int(stage_a_cfg.get("top_k_for_stage_b", 5)),
        "trials": completed_trials,
    }
    write_json(stage_a_summary, output_root / "stage_a_summary.json")
    return stage_a_summary


def run_stage_b(
    search_cfg: dict[str, Any],
    study_cfg: dict[str, Any],
    stage_a_summary: dict[str, Any],
    device: str,
    overwrite: bool,
) -> dict[str, Any]:
    base_experiment_path = resolve_project_path(search_cfg["base_experiment"], start=PROJECT_ROOT)
    stage_b_cfg = dict(search_cfg["stage_b"])
    output_root = resolve_project_path(stage_b_cfg["output_root"], start=PROJECT_ROOT)
    rerank_dir = output_root / "rerank_configs"

    if overwrite:
        _safe_rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rerank_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(seed) for seed in stage_b_cfg["seeds"]]
    top_k = int(stage_a_summary["top_k_for_stage_b"])
    std_penalty = float(stage_b_cfg.get("std_penalty", 0.25))

    candidate_trials = stage_a_summary["trials"][:top_k]
    rerank_records: list[dict[str, Any]] = []

    for candidate in candidate_trials:
        trial_number = int(candidate["trial_number"])
        params = dict(candidate["params"])
        experiment_name = f"{study_cfg['name']}_rerank_trial_{trial_number:03d}"

        overrides: dict[str, Any] = {"experiment": {"name": experiment_name}}
        for dotted_key, value in params.items():
            _set_nested(overrides, dotted_key, value)

        rerank_cfg_path = rerank_dir / f"rerank_trial_{trial_number:03d}.yaml"
        _materialize_experiment_config(
            base_experiment_path=base_experiment_path,
            overrides=overrides,
            output_path=rerank_cfg_path,
        )

        seed_metrics: list[dict[str, Any]] = []
        valid_values: list[float] = []
        test_values: list[float] = []

        for seed in seeds:
            run_payload = _run_train_text(
                experiment_config_path=rerank_cfg_path,
                output_root=output_root,
                seed=seed,
                device=device,
            )
            valid_macro_f1 = _extract_valid_macro_f1(run_payload["metrics"])
            test_macro_f1 = _extract_test_macro_f1(run_payload["metrics"])
            valid_values.append(valid_macro_f1)
            test_values.append(test_macro_f1)

            seed_metrics.append(
                {
                    "seed": seed,
                    "run_dir": run_payload["run_dir"],
                    "valid_macro_f1": valid_macro_f1,
                    "test_macro_f1": test_macro_f1,
                }
            )

        mean_valid = sum(valid_values) / len(valid_values)
        mean_test = sum(test_values) / len(test_values)
        std_valid = (
            math.sqrt(sum((x - mean_valid) ** 2 for x in valid_values) / len(valid_values))
            if len(valid_values) > 1
            else 0.0
        )
        robust_score = mean_valid - std_penalty * std_valid

        rerank_records.append(
            {
                "trial_number": trial_number,
                "params": params,
                "seed_metrics": seed_metrics,
                "mean_valid_macro_f1": mean_valid,
                "std_valid_macro_f1": std_valid,
                "mean_test_macro_f1": mean_test,
                "robust_score": robust_score,
            }
        )

    rerank_records.sort(key=lambda item: item["robust_score"], reverse=True)
    stage_b_summary = {
        "study_name": str(study_cfg["name"]) + "_stage_b",
        "std_penalty": std_penalty,
        "records": rerank_records,
        "best_record": rerank_records[0] if rerank_records else None,
    }
    write_json(stage_b_summary, output_root / "stage_b_summary.json")
    return stage_b_summary


def main() -> int:
    args = build_parser().parse_args()
    cfg_path = resolve_project_path(args.config, start=PROJECT_ROOT)
    config = load_yaml(cfg_path)

    study_cfg = dict(config["study"])
    search_cfg = dict(config["search"])
    space_cfg = dict(config["space"])

    stage_a_summary = None
    if args.stage in {"all", "stage_a"}:
        stage_a_summary = run_stage_a(
            search_cfg=search_cfg,
            study_cfg=study_cfg,
            space_cfg=space_cfg,
            device=args.device,
            overwrite=args.overwrite,
        )
        print("[OK] Stage A completed.")
        print(json.dumps(stage_a_summary, ensure_ascii=False, indent=2))

    if args.stage in {"all", "stage_b"}:
        if stage_a_summary is None:
            stage_a_output_root = resolve_project_path(search_cfg["stage_a"]["output_root"], start=PROJECT_ROOT)
            stage_a_summary_path = stage_a_output_root / "stage_a_summary.json"
            if not stage_a_summary_path.exists():
                raise FileNotFoundError(
                    "Stage B requested but stage_a_summary.json was not found. "
                    "Run stage_a first or use --stage all."
                )
            with stage_a_summary_path.open("r", encoding="utf-8") as f:
                stage_a_summary = json.load(f)

        stage_b_summary = run_stage_b(
            search_cfg=search_cfg,
            study_cfg=study_cfg,
            stage_a_summary=stage_a_summary,
            device=args.device,
            overwrite=args.overwrite,
        )
        print("[OK] Stage B completed.")
        print(json.dumps(stage_b_summary, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())