from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.io import load_expert_artifacts_and_prepare, save_torch_artifact
from src.utils.config import deep_update, load_yaml
from src.utils.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare fusion-ready artifacts from expert exports."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Fusion experiment config path.",
    )
    return parser


def resolve_experiment_config(config_path: Path) -> dict[str, Any]:
    exp_cfg = load_yaml(resolve_project_path(config_path, start=PROJECT_ROOT))
    defaults = exp_cfg.get("defaults", [])
    merged: dict[str, Any] = {}

    for entry in defaults:
        if not isinstance(entry, str):
            raise TypeError("Fusion config `defaults` entries must be string paths.")
        cfg_part = load_yaml(resolve_project_path(entry, start=PROJECT_ROOT))
        merged = deep_update(merged, cfg_part)

    exp_without_defaults = dict(exp_cfg)
    exp_without_defaults.pop("defaults", None)
    merged = deep_update(merged, exp_without_defaults)

    merged.setdefault("project", {})
    merged["project"].setdefault("root", str(PROJECT_ROOT))
    return merged


def _required_key(container: dict[str, Any], key: str, container_name: str) -> Any:
    if key not in container:
        raise KeyError(f"Missing required key `{key}` in {container_name}")
    return container[key]


def _artifact_path(root_dir: str | Path, split: str, project_root: str | Path) -> Path:
    return resolve_project_path(Path(root_dir) / f"{split}.pt", start=project_root)


def _resolve_optional_path(
    path_value: str | Path | None,
    project_root: str | Path,
) -> Path | None:
    if path_value is None or str(path_value).strip() == "":
        return None
    return resolve_project_path(path_value, start=project_root)


def _resolve_calibration_paths(
    calibration_cfg: dict[str, Any],
    project_root: str | Path,
) -> tuple[Path | None, Path | None]:
    required = bool(calibration_cfg.get("required", False))

    text_path = _resolve_optional_path(calibration_cfg.get("text"), project_root)
    speech_path = _resolve_optional_path(calibration_cfg.get("speech"), project_root)

    if required:
        if text_path is None:
            raise ValueError(
                "dataset.calibration.required=true but dataset.calibration.text is missing."
            )
        if speech_path is None:
            raise ValueError(
                "dataset.calibration.required=true but dataset.calibration.speech is missing."
            )
        if not text_path.exists():
            raise FileNotFoundError(f"Required text calibration file not found: {text_path}")
        if not speech_path.exists():
            raise FileNotFoundError(f"Required speech calibration file not found: {speech_path}")

    return text_path, speech_path


def main() -> int:
    args = build_parser().parse_args()

    config = resolve_experiment_config(args.config)
    project_root = config.get("project", {}).get("root", str(PROJECT_ROOT))

    dataset_cfg = _required_key(config, "dataset", "fusion config")
    reliability_cfg = dict(config.get("reliability", {}))

    text_export_dir = _required_key(dataset_cfg, "text_export_dir", "dataset")
    speech_export_dir = _required_key(dataset_cfg, "speech_export_dir", "dataset")
    fusion_artifacts_cfg = _required_key(dataset_cfg, "fusion_artifacts", "dataset")
    calibration_cfg = dict(dataset_cfg.get("calibration", {}))

    text_calibration_path, speech_calibration_path = _resolve_calibration_paths(
        calibration_cfg=calibration_cfg,
        project_root=project_root,
    )

    if text_calibration_path is not None:
        print(f"[INFO] Text calibration: {text_calibration_path}")
    else:
        print("[INFO] Text calibration: disabled")

    if speech_calibration_path is not None:
        print(f"[INFO] Speech calibration: {speech_calibration_path}")
    else:
        print("[INFO] Speech calibration: disabled")

    for split in ["train", "valid", "test"]:
        text_artifact_path = _artifact_path(text_export_dir, split, project_root)
        speech_artifact_path = _artifact_path(speech_export_dir, split, project_root)

        output_path = resolve_project_path(
            _required_key(fusion_artifacts_cfg, split, "dataset.fusion_artifacts"),
            start=project_root,
        )

        fusion_artifact = load_expert_artifacts_and_prepare(
            text_artifact_path=text_artifact_path,
            speech_artifact_path=speech_artifact_path,
            reliability_config=reliability_cfg,
            text_calibration_path=text_calibration_path,
            speech_calibration_path=speech_calibration_path,
        )
        save_torch_artifact(fusion_artifact, output_path)
        print(f"[OK] Prepared fusion artifact for split={split}: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())