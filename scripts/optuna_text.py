from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from copy import deepcopy
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
        description=(
            "Run Optuna hyperparameter search for the text branch without modifying "
            "the stable baseline."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Optuna search config YAML.")
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
        help="Device passed through to train_text.py: cpu, cuda, or auto.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete previous stage output directories before running.",
    )
    return parser


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
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
            return trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                step=float(step),
                log=log,
            )
        return trial.suggest_float(
            name,
            float(spec["low"]),
            float(spec["high"]),
            log=log,
        )

    if kind == "int":
        step = int(spec.get("step", 1))
        log = bool(spec.get("log", False))
        return trial.suggest_int(
            name,
            int(spec["low"]),
            int(spec["high"]),
            step=step,
            log=log,
        )

    raise KeyError(f"Unsupported search-space type: {kind}")


def _build_trial_overrides(trial: optuna.Trial, space_cfg: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for dotted_key, spec in space_cfg.items():
        value = _suggest_param(trial, dotted_key, dict(spec))
        _set_nested(overrides, dotted_key, value)
    return overrides


def _materialize_experiment_config(
    base_experiment_path: Path,
    overrides: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    base_cfg = load_yaml(base_experiment_path)
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
            f"Command: {' '.join(cmd)}\n\n"
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


def _extract_split_metric(metrics_payload: dict[str, Any], split: str, metric_name: str) -> float:
    split_section = metrics_payload.get(split)
    if not isinstance(split_section, dict):
        raise KeyError(f"`{split}` section not found in metrics.json")
    if metric_name not in split_section:
        raise KeyError(f"`{split}.{metric_name}` not found in metrics.json")
    return float(split_section[metric_name])


def _extract_metric(metrics_payload: dict[str, Any], metric_key: str) -> float:
    normalized = str(metric_key).strip()

    if "." in normalized:
        split, metric_name = normalized.split(".", maxsplit=1)
        return _extract_split_metric(metrics_payload, split=split, metric_name=metric_name)

    if "__" in normalized:
        split, metric_name = normalized.split("__", maxsplit=1)
        return _extract_split_metric(metrics_payload, split=split, metric_name=metric_name)

    for split in ("valid", "test", "train"):
        prefix = f"{split}_"
        if normalized.startswith(prefix):
            metric_name = normalized[len(prefix):]
            return _extract_split_metric(metrics_payload, split=split, metric_name=metric_name)

    raise KeyError(
        f"Unsupported objective metric key: {metric_key}. "
        "Use forms like valid_macro_f1, test_macro_f1, valid.accuracy, or test__weighted_f1."
    )


def _extract_valid_macro_f1(metrics_payload: dict[str, Any]) -> float:
    return _extract_split_metric(metrics_payload, split="valid", metric_name="macro_f1")


def _extract_test_macro_f1(metrics_payload: dict[str, Any]) -> float:
    return _extract_split_metric(metrics_payload, split="test", metric_name="macro_f1")


def _extract_best_train_macro_f1(metrics_payload: dict[str, Any]) -> float:
    fit_section = metrics_payload.get("fit", {})
    if not isinstance(fit_section, dict):
        return 0.0

    history = fit_section.get("history", [])
    if not isinstance(history, list):
        return 0.0

    values: list[float] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        train_metrics = item.get("train", {})
        if isinstance(train_metrics, dict) and "macro_f1" in train_metrics:
            values.append(float(train_metrics["macro_f1"]))

    return max(values) if values else 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _population_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return float(math.sqrt(variance))


def _seed_list(stage_cfg: dict[str, Any], fallback_seed: int = 42) -> list[int]:
    if "seeds" in stage_cfg:
        seeds = stage_cfg["seeds"]
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("`seeds` must be a non-empty list when provided.")
        return [int(seed) for seed in seeds]

    if "seed" in stage_cfg:
        return [int(stage_cfg["seed"])]

    return [int(fallback_seed)]


def _prepare_sampler(study_cfg: dict[str, Any]) -> optuna.samplers.BaseSampler:
    sampler_cfg = dict(study_cfg.get("sampler", {}))
    sampler_name = str(sampler_cfg.get("name", "tpe")).lower()

    if sampler_name != "tpe":
        raise KeyError(f"Unsupported sampler: {sampler_name}")

    return TPESampler(
        seed=int(sampler_cfg.get("seed", 2026)),
        multivariate=bool(sampler_cfg.get("multivariate", False)),
    )


def _prepare_pruner(study_cfg: dict[str, Any]) -> optuna.pruners.BasePruner:
    pruner_cfg = dict(study_cfg.get("pruner", {}))
    pruner_name = str(pruner_cfg.get("name", "median")).lower()

    if pruner_name == "none":
        return optuna.pruners.NopPruner()

    if pruner_name != "median":
        raise KeyError(f"Unsupported pruner: {pruner_name}")

    return MedianPruner(
        n_startup_trials=int(pruner_cfg.get("n_startup_trials", 8)),
        n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 1)),
        interval_steps=int(pruner_cfg.get("interval_steps", 1)),
    )


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _create_study(study_cfg: dict[str, Any], suffix: str) -> optuna.Study:
    storage = study_cfg.get("storage")
    return optuna.create_study(
        study_name=str(study_cfg["name"]) + suffix,
        direction=str(study_cfg.get("direction", "maximize")),
        sampler=_prepare_sampler(study_cfg),
        pruner=_prepare_pruner(study_cfg),
        storage=storage,
        load_if_exists=bool(storage),
    )


def _score_candidate(
    mean_objective: float,
    std_objective: float,
    mean_train_macro_f1: float,
    mean_valid_macro_f1: float,
    std_penalty: float,
    overfit_penalty: float,
) -> tuple[float, float]:
    overfit_gap = max(0.0, mean_train_macro_f1 - mean_valid_macro_f1)
    score = (
        mean_objective
        - float(std_penalty) * std_objective
        - float(overfit_penalty) * overfit_gap
    )
    return float(score), float(overfit_gap)


def _diagnostic_best_test_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [record for record in records if record.get("status", "complete") == "complete"]
    if not completed:
        return None
    return max(completed, key=lambda item: float(item.get("mean_test_macro_f1", -math.inf)))


def _write_stage_records(output_root: Path, stage_name: str, records: list[dict[str, Any]]) -> None:
    write_json(
        {
            "stage": stage_name,
            "records": records,
        },
        output_root / f"{stage_name}_records.json",
    )


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
    trials_dir = output_root / "artifacts" / "trial_configs"

    if overwrite:
        _safe_rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)

    study = _create_study(study_cfg, suffix="_stage_a")

    objective_metric = str(search_cfg.get("objective_metric", "valid_macro_f1"))
    seeds = _seed_list(stage_a_cfg)
    std_penalty = float(stage_a_cfg.get("std_penalty", 0.25))
    overfit_penalty = float(stage_a_cfg.get("overfit_penalty", 0.0))

    trial_records: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        overrides = _build_trial_overrides(trial, space_cfg)
        experiment_name = f"{study_cfg['name']}_trial_{trial.number:03d}"
        overrides = _deep_update(overrides, {"experiment": {"name": experiment_name}})

        trial_cfg_path = trials_dir / f"trial_{trial.number:03d}.yaml"
        materialized_cfg = _materialize_experiment_config(
            base_experiment_path=base_experiment_path,
            overrides=overrides,
            output_path=trial_cfg_path,
        )

        seed_metrics: list[dict[str, Any]] = []
        objective_values: list[float] = []
        train_macro_values: list[float] = []
        valid_macro_values: list[float] = []
        test_macro_values: list[float] = []

        record: dict[str, Any] = {
            "trial_number": trial.number,
            "status": "running",
            "experiment_name": materialized_cfg["experiment"]["name"],
            "config_path": str(trial_cfg_path.resolve()),
            "params": dict(trial.params),
            "seed_metrics": seed_metrics,
        }

        try:
            for seed_idx, seed in enumerate(seeds):
                run_payload = _run_train_text(
                    experiment_config_path=trial_cfg_path,
                    output_root=output_root,
                    seed=seed,
                    device=device,
                )

                metrics = run_payload["metrics"]
                objective_value = _extract_metric(metrics, objective_metric)
                train_macro_f1 = _extract_best_train_macro_f1(metrics)
                valid_macro_f1 = _extract_valid_macro_f1(metrics)
                test_macro_f1 = _extract_test_macro_f1(metrics)

                objective_values.append(objective_value)
                train_macro_values.append(train_macro_f1)
                valid_macro_values.append(valid_macro_f1)
                test_macro_values.append(test_macro_f1)

                mean_objective = _mean(objective_values)
                std_objective = _population_std(objective_values)
                mean_train = _mean(train_macro_values)
                mean_valid = _mean(valid_macro_values)
                partial_score, overfit_gap = _score_candidate(
                    mean_objective=mean_objective,
                    std_objective=std_objective,
                    mean_train_macro_f1=mean_train,
                    mean_valid_macro_f1=mean_valid,
                    std_penalty=std_penalty,
                    overfit_penalty=overfit_penalty,
                )

                seed_metrics.append(
                    {
                        "seed": seed,
                        "run_dir": run_payload["run_dir"],
                        "objective_metric": objective_metric,
                        "objective_value": objective_value,
                        "train_macro_f1_best": train_macro_f1,
                        "valid_macro_f1": valid_macro_f1,
                        "test_macro_f1": test_macro_f1,
                    }
                )

                trial.report(partial_score, step=seed_idx)
                if trial.should_prune():
                    record.update(
                        {
                            "status": "pruned",
                            "objective_metric": objective_metric,
                            "mean_objective": mean_objective,
                            "std_objective": std_objective,
                            "mean_train_macro_f1": mean_train,
                            "mean_valid_macro_f1": mean_valid,
                            "mean_test_macro_f1": _mean(test_macro_values),
                            "overfit_gap": overfit_gap,
                            "robust_score": partial_score,
                        }
                    )
                    trial_records.append(record)
                    _write_stage_records(output_root, "stage_a", trial_records)
                    raise optuna.TrialPruned()

            mean_objective = _mean(objective_values)
            std_objective = _population_std(objective_values)
            mean_train = _mean(train_macro_values)
            mean_valid = _mean(valid_macro_values)
            mean_test = _mean(test_macro_values)
            robust_score, overfit_gap = _score_candidate(
                mean_objective=mean_objective,
                std_objective=std_objective,
                mean_train_macro_f1=mean_train,
                mean_valid_macro_f1=mean_valid,
                std_penalty=std_penalty,
                overfit_penalty=overfit_penalty,
            )

            record.update(
                {
                    "status": "complete",
                    "objective_metric": objective_metric,
                    "mean_objective": mean_objective,
                    "std_objective": std_objective,
                    "robust_score": robust_score,
                    "mean_train_macro_f1": mean_train,
                    "std_train_macro_f1": _population_std(train_macro_values),
                    "mean_valid_macro_f1": mean_valid,
                    "std_valid_macro_f1": _population_std(valid_macro_values),
                    "mean_test_macro_f1": mean_test,
                    "std_test_macro_f1": _population_std(test_macro_values),
                    "overfit_gap": overfit_gap,
                    "valid_test_gap": mean_valid - mean_test,
                }
            )

            trial.set_user_attr("config_path", str(trial_cfg_path.resolve()))
            trial.set_user_attr("robust_score", robust_score)
            trial.set_user_attr("mean_valid_macro_f1", mean_valid)
            trial.set_user_attr("mean_test_macro_f1", mean_test)
            trial.set_user_attr("overfit_gap", overfit_gap)

            trial_records.append(record)
            _write_stage_records(output_root, "stage_a", trial_records)
            return robust_score

        except optuna.TrialPruned:
            raise
        except Exception as exc:
            record.update({"status": "failed", "error": str(exc)})
            trial_records.append(record)
            _write_stage_records(output_root, "stage_a", trial_records)
            raise

    study.optimize(
        objective,
        n_trials=int(stage_a_cfg["n_trials"]),
        catch=(RuntimeError,),
    )

    completed_records = [record for record in trial_records if record.get("status") == "complete"]
    completed_records.sort(key=lambda item: float(item["robust_score"]), reverse=True)

    if not completed_records:
        raise RuntimeError("Stage A finished without any completed trials.")

    stage_a_summary = {
        "study_name": str(study_cfg["name"]) + "_stage_a",
        "objective_metric": objective_metric,
        "seeds": seeds,
        "std_penalty": std_penalty,
        "overfit_penalty": overfit_penalty,
        "best_trial_number": int(completed_records[0]["trial_number"]),
        "best_value": float(completed_records[0]["robust_score"]),
        "best_params": dict(completed_records[0]["params"]),
        "top_k_for_stage_b": int(stage_a_cfg.get("top_k_for_stage_b", 5)),
        "trials": completed_records,
        "diagnostic_best_test_record": _diagnostic_best_test_record(completed_records),
        "num_trials_total": len(trial_records),
        "num_trials_complete": len(completed_records),
        "num_trials_pruned": len([r for r in trial_records if r.get("status") == "pruned"]),
        "num_trials_failed": len([r for r in trial_records if r.get("status") == "failed"]),
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

    objective_metric = str(search_cfg.get("objective_metric", "valid_macro_f1"))
    seeds = _seed_list(stage_b_cfg)
    top_k = int(stage_a_summary["top_k_for_stage_b"])
    std_penalty = float(stage_b_cfg.get("std_penalty", 0.50))
    overfit_penalty = float(stage_b_cfg.get("overfit_penalty", 0.0))

    candidate_trials = list(stage_a_summary["trials"])[:top_k]
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
        objective_values: list[float] = []
        train_macro_values: list[float] = []
        valid_macro_values: list[float] = []
        test_macro_values: list[float] = []

        for seed in seeds:
            run_payload = _run_train_text(
                experiment_config_path=rerank_cfg_path,
                output_root=output_root,
                seed=seed,
                device=device,
            )

            metrics = run_payload["metrics"]
            objective_value = _extract_metric(metrics, objective_metric)
            train_macro_f1 = _extract_best_train_macro_f1(metrics)
            valid_macro_f1 = _extract_valid_macro_f1(metrics)
            test_macro_f1 = _extract_test_macro_f1(metrics)

            objective_values.append(objective_value)
            train_macro_values.append(train_macro_f1)
            valid_macro_values.append(valid_macro_f1)
            test_macro_values.append(test_macro_f1)

            seed_metrics.append(
                {
                    "seed": seed,
                    "run_dir": run_payload["run_dir"],
                    "objective_metric": objective_metric,
                    "objective_value": objective_value,
                    "train_macro_f1_best": train_macro_f1,
                    "valid_macro_f1": valid_macro_f1,
                    "test_macro_f1": test_macro_f1,
                }
            )

        mean_objective = _mean(objective_values)
        std_objective = _population_std(objective_values)
        mean_train = _mean(train_macro_values)
        mean_valid = _mean(valid_macro_values)
        mean_test = _mean(test_macro_values)
        robust_score, overfit_gap = _score_candidate(
            mean_objective=mean_objective,
            std_objective=std_objective,
            mean_train_macro_f1=mean_train,
            mean_valid_macro_f1=mean_valid,
            std_penalty=std_penalty,
            overfit_penalty=overfit_penalty,
        )

        rerank_records.append(
            {
                "trial_number": trial_number,
                "params": params,
                "config_path": str(rerank_cfg_path.resolve()),
                "seed_metrics": seed_metrics,
                "objective_metric": objective_metric,
                "mean_objective": mean_objective,
                "std_objective": std_objective,
                "robust_score": robust_score,
                "mean_train_macro_f1": mean_train,
                "std_train_macro_f1": _population_std(train_macro_values),
                "mean_valid_macro_f1": mean_valid,
                "std_valid_macro_f1": _population_std(valid_macro_values),
                "mean_test_macro_f1": mean_test,
                "std_test_macro_f1": _population_std(test_macro_values),
                "overfit_gap": overfit_gap,
                "valid_test_gap": mean_valid - mean_test,
            }
        )

        write_json(
            {
                "stage": "stage_b",
                "records": rerank_records,
            },
            output_root / "stage_b_records.json",
        )

    rerank_records.sort(key=lambda item: float(item["robust_score"]), reverse=True)

    stage_b_summary = {
        "study_name": str(study_cfg["name"]) + "_stage_b",
        "objective_metric": objective_metric,
        "seeds": seeds,
        "std_penalty": std_penalty,
        "overfit_penalty": overfit_penalty,
        "records": rerank_records,
        "best_record": rerank_records[0] if rerank_records else None,
        "diagnostic_best_test_record": _diagnostic_best_test_record(rerank_records),
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

    stage_a_cfg = dict(search_cfg.get("stage_a", {}))
    stage_b_cfg = dict(search_cfg.get("stage_b", {}))

    stage_a_summary = None

    if args.stage in {"all", "stage_a"}:
        if not bool(stage_a_cfg.get("enabled", True)):
            print("[SKIP] Stage A is disabled in config.")
        else:
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
        if not bool(stage_b_cfg.get("enabled", True)):
            print("[SKIP] Stage B is disabled in config.")
            return 0

        if stage_a_summary is None:
            stage_a_output_root = resolve_project_path(
                search_cfg["stage_a"]["output_root"],
                start=PROJECT_ROOT,
            )
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