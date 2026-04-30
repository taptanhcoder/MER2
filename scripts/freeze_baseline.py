from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml
from src.utils.io import write_json, write_yaml
from src.utils.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the current text/speech/fusion baselines into an immutable "
            "snapshot directory for reproducible research."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Baseline registry config path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing baseline snapshot directory.",
    )
    return parser


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def list_directory_files(path: Path) -> list[Path]:
    return sorted([p for p in path.rglob("*") if p.is_file()])


def hash_path(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "type": "file",
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    if path.is_dir():
        files = list_directory_files(path)
        return {
            "type": "directory",
            "num_files": len(files),
            "files": {
                str(file.relative_to(path)): {
                    "sha256": sha256_file(file),
                    "size_bytes": file.stat().st_size,
                }
                for file in files
            },
        }

    raise FileNotFoundError(f"Cannot hash missing path: {path}")


def safe_copy_path(src: Path, dst: Path) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return

    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return

    raise FileNotFoundError(f"Cannot copy missing path: {src}")


def read_test_metrics(metrics_json_path: Path) -> dict[str, Any]:
    with metrics_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if "test" not in payload:
        raise KeyError(f"`test` section not found in metrics file: {metrics_json_path}")
    return payload["test"]


def verify_expected_metrics(
    actual: dict[str, Any],
    expected: dict[str, Any],
    tolerance: float,
    component_name: str,
) -> None:
    for key, expected_value in expected.items():
        if key not in actual:
            raise KeyError(f"Metric `{key}` missing in {component_name} actual test metrics")
        actual_value = float(actual[key])
        if abs(actual_value - float(expected_value)) > tolerance:
            raise ValueError(
                f"{component_name} metric mismatch for `{key}`: "
                f"expected={expected_value}, actual={actual_value}, tolerance={tolerance}"
            )


def copy_component(
    component_name: str,
    component_cfg: dict[str, Any],
    baseline_root: Path,
    tolerance: float,
) -> dict[str, Any]:
    run_dir = resolve_project_path(component_cfg["run_dir"], start=PROJECT_ROOT)
    if not run_dir.exists():
        raise FileNotFoundError(f"{component_name} run_dir not found: {run_dir}")

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{component_name} metrics.json not found: {metrics_path}")

    actual_test_metrics = read_test_metrics(metrics_path)
    expected_test_metrics = dict(component_cfg.get("expected_test_metrics", {}))
    if expected_test_metrics:
        verify_expected_metrics(
            actual=actual_test_metrics,
            expected=expected_test_metrics,
            tolerance=tolerance,
            component_name=component_name,
        )

    component_root = baseline_root / component_name
    component_root.mkdir(parents=True, exist_ok=True)

    include_entries = list(component_cfg.get("include", []))
    copied_entries: dict[str, Any] = {}

    for entry in include_entries:
        src = run_dir / entry
        if not src.exists():
            raise FileNotFoundError(f"{component_name} required artifact missing: {src}")

        dst = component_root / entry
        safe_copy_path(src, dst)
        copied_entries[entry] = {
            "source": str(src.resolve()),
            "snapshot": str(dst.resolve()),
            "hash": hash_path(dst),
        }

    return {
        "run_dir": str(run_dir.resolve()),
        "expected_test_metrics": expected_test_metrics,
        "actual_test_metrics": actual_test_metrics,
        "copied_entries": copied_entries,
    }


def copy_shared_artifacts(shared_cfg: dict[str, Any], baseline_root: Path) -> dict[str, Any]:
    shared_root = baseline_root / "shared_artifacts"
    shared_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {}
    for key, rel_path in shared_cfg.items():
        src = resolve_project_path(rel_path, start=PROJECT_ROOT)
        if not src.exists():
            raise FileNotFoundError(f"Shared artifact path not found for `{key}`: {src}")

        dst = shared_root / key
        safe_copy_path(src, dst)
        manifest[key] = {
            "source": str(src.resolve()),
            "snapshot": str(dst.resolve()),
            "hash": hash_path(dst),
        }

    return manifest


def main() -> int:
    args = build_parser().parse_args()

    cfg_path = resolve_project_path(args.config, start=PROJECT_ROOT)
    config = load_yaml(cfg_path)

    baseline_cfg = dict(config["baseline"])
    baseline_name = str(baseline_cfg["name"])
    baseline_root = resolve_project_path(baseline_cfg["output_root"], start=PROJECT_ROOT)
    tolerance = float(dict(baseline_cfg.get("verification", {})).get("metric_tolerance", 1.0e-8))

    if baseline_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Baseline snapshot already exists: {baseline_root}. "
                "Use --overwrite if you intentionally want to recreate it."
            )
        shutil.rmtree(baseline_root)

    baseline_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "baseline_name": baseline_name,
        "description": baseline_cfg.get("description", ""),
        "source_config": str(cfg_path.resolve()),
        "snapshot_root": str(baseline_root.resolve()),
        "components": {},
        "shared_artifacts": {},
    }

    for component_name in ["text", "speech", "fusion"]:
        component_cfg = dict(baseline_cfg[component_name])
        manifest["components"][component_name] = copy_component(
            component_name=component_name,
            component_cfg=component_cfg,
            baseline_root=baseline_root,
            tolerance=tolerance,
        )

    manifest["shared_artifacts"] = copy_shared_artifacts(
        shared_cfg=dict(baseline_cfg.get("shared_artifacts", {})),
        baseline_root=baseline_root,
    )

    write_yaml(config, baseline_root / "baseline_config_resolved.yaml")
    write_json(manifest, baseline_root / "baseline_manifest.json")

    print(f"[OK] Baseline snapshot created: {baseline_root}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())