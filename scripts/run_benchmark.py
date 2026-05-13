from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.results_summary import (
    resolve_experiment_name_from_config,
    summarize_from_run_dirs,
)
from src.utils.config import load_yaml
from src.utils.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run benchmark experiments across multiple seeds and summarize results."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to benchmark config YAML.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Pass --debug to text/speech training scripts. "
            "Fusion benchmark does not use --debug because run_fusion.py has no debug flag."
        ),
    )
    return parser


def _validate_benchmark_config(benchmark_cfg: Any) -> dict[str, Any]:
    if not isinstance(benchmark_cfg, dict):
        raise KeyError(
            "Benchmark config must contain a top-level `benchmark` mapping. "
            "Expected keys: benchmark.task, benchmark.experiments, benchmark.seeds."
        )

    required_keys = ["task", "experiments", "seeds"]
    missing_keys = [key for key in required_keys if key not in benchmark_cfg]
    if missing_keys:
        raise KeyError(
            f"Missing required benchmark config keys: {missing_keys}. "
            "These keys must be nested under the top-level `benchmark` key."
        )

    experiments = benchmark_cfg.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("benchmark.experiments must be a non-empty list.")

    seeds = benchmark_cfg.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("benchmark.seeds must be a non-empty list.")

    return benchmark_cfg


def _resolve_train_script(task: str) -> Path:
    if task == "text":
        return resolve_project_path("scripts/train_text.py", start=PROJECT_ROOT)

    if task == "speech":
        return resolve_project_path("scripts/train_speech.py", start=PROJECT_ROOT)

    if task == "fusion":
        return resolve_project_path("scripts/run_fusion.py", start=PROJECT_ROOT)

    raise ValueError(
        f"Unsupported benchmark task: {task}. "
        "Supported tasks are: text, speech, fusion."
    )


def _build_command(
    train_script: Path,
    experiment_config_path: Path,
    seed: int,
    output_root: Path,
    device: str | None,
    task: str,
    debug: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(train_script),
        "--config",
        str(experiment_config_path),
        "--seed",
        str(seed),
        "--output-root",
        str(output_root),
    ]

    if device is not None:
        cmd.extend(["--device", str(device)])

    if debug and task in {"text", "speech"}:
        cmd.append("--debug")

    return cmd


def main() -> int:
    args = build_parser().parse_args()

    config_path = resolve_project_path(args.config, start=PROJECT_ROOT)
    cfg = load_yaml(config_path)
    benchmark_cfg = _validate_benchmark_config(cfg.get("benchmark"))

    task = str(benchmark_cfg["task"])
    experiments = list(benchmark_cfg["experiments"])
    seeds = [int(seed) for seed in benchmark_cfg["seeds"]]
    device = benchmark_cfg.get("device", None)

    output_root = resolve_project_path(
        benchmark_cfg.get("output_root", "outputs/runs"),
        start=PROJECT_ROOT,
    )
    report_dir = resolve_project_path(
        benchmark_cfg.get("report_dir", "outputs/reports/benchmark"),
        start=PROJECT_ROOT,
    )
    skip_existing = bool(benchmark_cfg.get("skip_existing", True))

    train_script = _resolve_train_script(task)
    run_dirs: list[Path] = []

    for experiment_config in experiments:
        experiment_config_path = resolve_project_path(experiment_config, start=PROJECT_ROOT)
        experiment_name = resolve_experiment_name_from_config(experiment_config_path)

        for seed in seeds:
            run_dir = output_root / experiment_name / f"seed_{seed}"
            metrics_path = run_dir / "metrics.json"
            run_dirs.append(run_dir)

            if skip_existing and metrics_path.exists():
                print(f"[SKIP] Existing run found: {run_dir}")
                continue

            cmd = _build_command(
                train_script=train_script,
                experiment_config_path=experiment_config_path,
                seed=seed,
                output_root=output_root,
                device=device,
                task=task,
                debug=args.debug,
            )

            print("[RUN]", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    per_run_df, aggregate_df = summarize_from_run_dirs(run_dirs, report_dir=report_dir)

    print(f"[OK] Wrote per-run summary to: {report_dir / 'per_run_metrics.csv'}")
    print(f"[OK] Wrote aggregate summary to: {report_dir / 'aggregate_metrics.csv'}")

    stability_path = report_dir / "stability_summary.csv"
    if stability_path.exists():
        print(f"[OK] Wrote stability summary to: {stability_path}")

    if not aggregate_df.empty:
        print()
        print(aggregate_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())