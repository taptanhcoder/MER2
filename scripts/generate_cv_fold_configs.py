from __future__ import annotations

import argparse
import json
import shlex
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml
from src.utils.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fold/seed-specific configs for CV evaluation without "
            "modifying fixed-split baseline/final configs."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/cv/final_light_bica_gate_cv.yaml"),
        help="CV config generator YAML.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default=None,
        help="Generate only one fold, e.g. fold_00. If omitted, generate all folds.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Generate only one seed. If omitted, generate all configured seeds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated configs if they already exist.",
    )
    return parser


def _resolve(path: str | Path) -> Path:
    return resolve_project_path(path, start=PROJECT_ROOT)


def _as_posix(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def _safe_write_yaml(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing generated config: {path}. "
            "Pass --overwrite to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _safe_write_text(path: Path, text: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing generated file: {path}. "
            "Pass --overwrite to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _set_path(cfg: dict[str, Any], dotted_path: str, value: Any) -> None:
    node: dict[str, Any] = cfg
    parts = dotted_path.split(".")

    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child

    node[parts[-1]] = value


def _delete_path(cfg: dict[str, Any], dotted_path: str) -> None:
    node: Any = cfg
    parts = dotted_path.split(".")

    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        node = node[part]

    if isinstance(node, dict):
        node.pop(parts[-1], None)


def _set_existing_or_create(
    cfg: dict[str, Any],
    dotted_paths: list[str],
    value: Any,
) -> None:
    found = False

    for dotted_path in dotted_paths:
        node: Any = cfg
        parts = dotted_path.split(".")
        exists = True

        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                exists = False
                break
            node = node[part]

        if exists and isinstance(node, dict) and parts[-1] in node:
            node[parts[-1]] = value
            found = True

    if not found and dotted_paths:
        _set_path(cfg, dotted_paths[0], value)


def _is_forbidden_value(value: str, forbidden_substrings: list[str]) -> bool:
    normalized = _as_posix(value)
    return any(item in normalized for item in forbidden_substrings)


def _purge_forbidden_values(obj: Any, forbidden_substrings: list[str]) -> Any:
    """
    Remove inherited fixed-split references from generated CV configs.

    This prevents CV configs from reading:
    - fixed-split fusion artifacts
    - fixed-split fusion exports
    - fixed-split temperature files
    """
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for key, value in obj.items():
            if isinstance(value, str) and _is_forbidden_value(value, forbidden_substrings):
                continue

            cleaned_value = _purge_forbidden_values(value, forbidden_substrings)
            if cleaned_value is not None:
                cleaned[key] = cleaned_value

        return cleaned

    if isinstance(obj, list):
        cleaned_items = []
        for item in obj:
            if isinstance(item, str) and _is_forbidden_value(item, forbidden_substrings):
                continue

            cleaned_item = _purge_forbidden_values(item, forbidden_substrings)
            if cleaned_item is not None:
                cleaned_items.append(cleaned_item)

        return cleaned_items

    return obj


def _purge_legacy_fusion_keys(cfg: dict[str, Any]) -> None:
    """
    Remove common path containers from source fusion config before re-injecting
    fold-specific CV paths.
    """
    for path in [
        "text_export_dir",
        "speech_export_dir",
        "fusion_artifact_dir",
        "artifact_dir",
        "calibration_paths",
        "temperature_paths",
        "dataset.fusion_artifacts",
        "dataset.train_artifact",
        "dataset.valid_artifact",
        "dataset.test_artifact",
        "dataset.train",
        "dataset.valid",
        "dataset.test",
        "artifacts.train",
        "artifacts.valid",
        "artifacts.test",
        "artifacts.train_artifact",
        "artifacts.valid_artifact",
        "artifacts.test_artifact",
        "fusion_artifacts.train",
        "fusion_artifacts.valid",
        "fusion_artifacts.test",
        "fusion_artifacts.train_artifact",
        "fusion_artifacts.valid_artifact",
        "fusion_artifacts.test_artifact",
        "paths.train",
        "paths.valid",
        "paths.test",
        "paths.train_artifact",
        "paths.valid_artifact",
        "paths.test_artifact",
    ]:
        _delete_path(cfg, path)


def _assert_no_forbidden_values(
    cfg: dict[str, Any],
    forbidden_substrings: list[str],
    config_name: str,
) -> None:
    hits: list[str] = []

    def visit(value: Any, prefix: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                visit(child, f"{prefix}[{idx}]")
        elif isinstance(value, str) and _is_forbidden_value(value, forbidden_substrings):
            hits.append(f"{prefix}: {value}")

    visit(cfg, "")

    if hits:
        preview = "\n".join(hits[:20])
        raise ValueError(
            f"Generated CV config `{config_name}` still contains fixed-split paths:\n"
            f"{preview}"
        )


def _assert_prepare_fusion_artifact_schema(
    cfg: dict[str, Any],
    config_name: str,
) -> None:
    """
    Validate the exact schema required by scripts/prepare_fusion_artifacts.py.

    This catches missing keys at config-generation time instead of failing only
    after the expensive text/speech train-export stages.
    """
    dataset = cfg.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"{config_name} missing required mapping `dataset`.")

    required_dataset_keys = [
        "text_export_dir",
        "speech_export_dir",
        "output_dir",
        "fusion_artifacts",
        "calibration",
    ]
    missing = [key for key in required_dataset_keys if key not in dataset]
    if missing:
        raise ValueError(f"{config_name} missing dataset keys: {missing}")

    fusion_artifacts = dataset.get("fusion_artifacts")
    if not isinstance(fusion_artifacts, dict):
        raise ValueError(f"{config_name} `dataset.fusion_artifacts` must be a mapping.")

    for split in ["train", "valid", "test"]:
        if split not in fusion_artifacts:
            raise ValueError(
                f"{config_name} missing dataset.fusion_artifacts.{split}"
            )

    calibration = dataset.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError(f"{config_name} `dataset.calibration` must be a mapping.")

    if calibration.get("required", False):
        for key in ["text", "speech"]:
            if key not in calibration:
                raise ValueError(
                    f"{config_name} missing dataset.calibration.{key}"
                )


def _append_cv_metadata(
    cfg: dict[str, Any],
    fold: str,
    seed: int,
    role: str,
    source_config: str,
) -> None:
    cfg.setdefault("cv", {})
    cfg["cv"].update(
        {
            "enabled": True,
            "fold": fold,
            "seed": int(seed),
            "role": role,
            "source_config_name": Path(source_config).name,
        }
    )


def _load_source_config(path: str | Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Source config not found: {resolved}")

    cfg = load_yaml(resolved)
    if not isinstance(cfg, dict):
        raise TypeError(f"Source config must be a YAML mapping: {resolved}")

    return cfg


def _fold_paths(folds_dir: Path, fold: str) -> dict[str, str]:
    fold_dir = folds_dir / fold
    paths = {
        "train": fold_dir / "train.csv",
        "valid": fold_dir / "valid.csv",
        "test": fold_dir / "test.csv",
    }

    for split, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} CSV for {fold}: {path}")

    return {split: _as_posix(path) for split, path in paths.items()}


def _apply_dataset_paths(
    cfg: dict[str, Any],
    split_paths: dict[str, str],
    cv_cfg: dict[str, Any],
) -> None:
    _set_existing_or_create(cfg, ["dataset.train_csv"], split_paths["train"])
    _set_existing_or_create(cfg, ["dataset.valid_csv"], split_paths["valid"])
    _set_existing_or_create(cfg, ["dataset.test_csv"], split_paths["test"])

    _set_existing_or_create(cfg, ["dataset.id_col"], cv_cfg.get("id_col", "sample_id"))
    _set_existing_or_create(cfg, ["dataset.label_col"], cv_cfg.get("label_col", "label"))
    _set_existing_or_create(
        cfg,
        ["dataset.label_id_col"],
        cv_cfg.get("label_id_col", "label_id"),
    )
    _set_existing_or_create(cfg, ["dataset.text_col"], cv_cfg.get("text_col", "text"))
    _set_existing_or_create(
        cfg,
        ["dataset.audio_path_col"],
        cv_cfg.get("audio_path_col", "audio_path"),
    )
    _set_existing_or_create(
        cfg,
        ["dataset.group_id_col"],
        cv_cfg.get("group_id_col", "group_id"),
    )


def _apply_seed(cfg: dict[str, Any], seed: int) -> None:
    for path in [
        "seed",
        "runtime.seed",
        "train.seed",
        "training.seed",
        "experiment.seed",
    ]:
        _set_path(cfg, path, int(seed))


def _apply_experiment_identity(
    cfg: dict[str, Any],
    experiment_name: str,
    run_dir: str,
    output_root: str,
) -> None:
    _set_existing_or_create(
        cfg,
        [
            "experiment.name",
            "name",
            "run.name",
            "experiment_name",
        ],
        experiment_name,
    )

    cfg["experiment_name"] = experiment_name

    for path in [
        "output.run_dir",
        "output_dir",
        "run_dir",
        "outputs.run_dir",
        "train.output_dir",
        "runtime.output_dir",
    ]:
        _set_path(cfg, path, run_dir)

    for path in [
        "output.root",
        "output_root",
        "outputs.root",
        "runtime.output_root",
    ]:
        _set_path(cfg, path, output_root)


def _prepare_text_config(
    source_cfg: dict[str, Any],
    source_config_path: str,
    split_paths: dict[str, str],
    cv_cfg: dict[str, Any],
    names_cfg: dict[str, Any],
    fold: str,
    seed: int,
    output_root: Path,
    forbidden_substrings: list[str],
) -> tuple[dict[str, Any], str]:
    cfg = deepcopy(source_cfg)

    experiment = str(names_cfg.get("text_experiment", "text_phobert_headtail128"))
    experiment_cv = f"{experiment}_{fold}_seed_{seed}"

    actual_run_dir = Path("outputs") / "runs" / "text" / experiment_cv / f"seed_{seed}"

    _apply_dataset_paths(cfg, split_paths, cv_cfg)
    _apply_seed(cfg, seed)

    _set_existing_or_create(
        cfg,
        [
            "experiment.name",
            "name",
            "run.name",
            "experiment_name",
        ],
        experiment_cv,
    )

    cfg["experiment_name"] = experiment_cv
    cfg["seed"] = int(seed)

    _append_cv_metadata(cfg, fold, seed, "text", source_config_path)
    cfg["cv"]["actual_run_dir"] = _as_posix(actual_run_dir)

    _assert_no_forbidden_values(
        cfg,
        forbidden_substrings,
        config_name=f"text:{fold}:seed_{seed}",
    )

    return cfg, _as_posix(actual_run_dir)


def _prepare_speech_config(
    source_cfg: dict[str, Any],
    source_config_path: str,
    split_paths: dict[str, str],
    cv_cfg: dict[str, Any],
    names_cfg: dict[str, Any],
    fold: str,
    seed: int,
    output_root: Path,
    forbidden_substrings: list[str],
) -> tuple[dict[str, Any], str]:
    cfg = deepcopy(source_cfg)

    experiment = str(names_cfg.get("speech_experiment", "speech_hubert_ce"))
    experiment_cv = f"{experiment}_{fold}_seed_{seed}"

    actual_run_dir = Path("outputs") / "runs" / "speech" / experiment_cv / f"seed_{seed}"

    _apply_dataset_paths(cfg, split_paths, cv_cfg)
    _apply_seed(cfg, seed)

    _set_existing_or_create(
        cfg,
        [
            "experiment.name",
            "name",
            "run.name",
            "experiment_name",
        ],
        experiment_cv,
    )

    cfg["experiment_name"] = experiment_cv
    cfg["seed"] = int(seed)

    _append_cv_metadata(cfg, fold, seed, "speech", source_config_path)
    cfg["cv"]["actual_run_dir"] = _as_posix(actual_run_dir)

    _assert_no_forbidden_values(
        cfg,
        forbidden_substrings,
        config_name=f"speech:{fold}:seed_{seed}",
    )

    return cfg, _as_posix(actual_run_dir)


def _apply_fusion_artifact_paths(
    cfg: dict[str, Any],
    artifact_dir: Path,
) -> None:
    train_artifact = _as_posix(artifact_dir / "train.pt")
    valid_artifact = _as_posix(artifact_dir / "valid.pt")
    test_artifact = _as_posix(artifact_dir / "test.pt")

    # Required schema for scripts/run_fusion.py:
    # cfg["dataset"]["fusion_artifacts"]["train"|"valid"|"test"]
    _set_path(cfg, "dataset.fusion_artifacts.train", train_artifact)
    _set_path(cfg, "dataset.fusion_artifacts.valid", valid_artifact)
    _set_path(cfg, "dataset.fusion_artifacts.test", test_artifact)

    # Compatibility aliases for older/newer config readers.
    for base in ["dataset", "artifacts", "fusion_artifacts", "paths"]:
        _set_path(cfg, f"{base}.train_artifact", train_artifact)
        _set_path(cfg, f"{base}.valid_artifact", valid_artifact)
        _set_path(cfg, f"{base}.test_artifact", test_artifact)
        _set_path(cfg, f"{base}.train", train_artifact)
        _set_path(cfg, f"{base}.valid", valid_artifact)
        _set_path(cfg, f"{base}.test", test_artifact)


def _prepare_fusion_artifact_config(
    source_cfg: dict[str, Any],
    source_config_path: str,
    fold: str,
    seed: int,
    text_run_dir: str,
    speech_run_dir: str,
    output_root: Path,
    text_export_dir: Path,
    speech_export_dir: Path,
    fusion_artifact_dir: Path,
    forbidden_substrings: list[str],
) -> dict[str, Any]:
    cfg = deepcopy(source_cfg)

    cfg = _purge_forbidden_values(cfg, forbidden_substrings)
    _purge_legacy_fusion_keys(cfg)

    train_artifact = fusion_artifact_dir / "train.pt"
    valid_artifact = fusion_artifact_dir / "valid.pt"
    test_artifact = fusion_artifact_dir / "test.pt"

    text_temperature = Path(text_run_dir) / "temperature.json"
    speech_temperature = Path(speech_run_dir) / "temperature.json"

    # Required schema for scripts/prepare_fusion_artifacts.py
    _set_path(cfg, "dataset.text_export_dir", _as_posix(text_export_dir))
    _set_path(cfg, "dataset.speech_export_dir", _as_posix(speech_export_dir))
    _set_path(cfg, "dataset.output_dir", _as_posix(fusion_artifact_dir))

    # Required by scripts/prepare_fusion_artifacts.py:
    # dataset.fusion_artifacts.{train,valid,test}
    _set_path(cfg, "dataset.fusion_artifacts.train", _as_posix(train_artifact))
    _set_path(cfg, "dataset.fusion_artifacts.valid", _as_posix(valid_artifact))
    _set_path(cfg, "dataset.fusion_artifacts.test", _as_posix(test_artifact))

    # Required by scripts/prepare_fusion_artifacts.py when
    # dataset.calibration.required=true.
    _set_path(cfg, "dataset.calibration.required", True)
    _set_path(cfg, "dataset.calibration.text", _as_posix(text_temperature))
    _set_path(cfg, "dataset.calibration.speech", _as_posix(speech_temperature))

    # Compatibility aliases for other fusion scripts/config readers.
    _set_path(cfg, "output_dir", _as_posix(fusion_artifact_dir))
    _set_path(cfg, "artifact_dir", _as_posix(fusion_artifact_dir))
    _set_path(cfg, "fusion_artifact_dir", _as_posix(fusion_artifact_dir))
    _set_path(cfg, "artifacts.output_dir", _as_posix(fusion_artifact_dir))
    _set_path(cfg, "fusion_artifacts.output_dir", _as_posix(fusion_artifact_dir))

    _set_path(cfg, "artifacts.train", _as_posix(train_artifact))
    _set_path(cfg, "artifacts.valid", _as_posix(valid_artifact))
    _set_path(cfg, "artifacts.test", _as_posix(test_artifact))

    _set_path(cfg, "fusion_artifacts.train", _as_posix(train_artifact))
    _set_path(cfg, "fusion_artifacts.valid", _as_posix(valid_artifact))
    _set_path(cfg, "fusion_artifacts.test", _as_posix(test_artifact))

    # Additional top-level aliases for inspection and compatibility.
    _set_path(cfg, "calibration.text", _as_posix(text_temperature))
    _set_path(cfg, "calibration.speech", _as_posix(speech_temperature))

    for base in ["text", "experts.text", "text_expert"]:
        _set_path(cfg, f"{base}.run_dir", text_run_dir)
        _set_path(cfg, f"{base}.export_dir", _as_posix(text_export_dir))
        _set_path(
            cfg,
            f"{base}.calibration_path",
            _as_posix(text_temperature),
        )

    for base in ["speech", "experts.speech", "speech_expert"]:
        _set_path(cfg, f"{base}.run_dir", speech_run_dir)
        _set_path(cfg, f"{base}.export_dir", _as_posix(speech_export_dir))
        _set_path(
            cfg,
            f"{base}.calibration_path",
            _as_posix(speech_temperature),
        )

    _apply_fusion_artifact_paths(cfg, fusion_artifact_dir)
    _append_cv_metadata(cfg, fold, seed, "prepare_fusion_artifacts", source_config_path)

    _assert_prepare_fusion_artifact_schema(
        cfg,
        config_name=f"prepare_fusion_artifacts:{fold}:seed_{seed}",
    )
    _assert_no_forbidden_values(
        cfg,
        forbidden_substrings,
        config_name=f"prepare_fusion_artifacts:{fold}:seed_{seed}",
    )

    return cfg


def _prepare_fusion_train_config(
    source_cfg: dict[str, Any],
    source_config_path: str,
    names_cfg: dict[str, Any],
    fold: str,
    seed: int,
    output_root: Path,
    fusion_artifact_dir: Path,
    forbidden_substrings: list[str],
) -> tuple[dict[str, Any], str]:
    cfg = deepcopy(source_cfg)

    cfg = _purge_forbidden_values(cfg, forbidden_substrings)
    _purge_legacy_fusion_keys(cfg)

    experiment = str(names_cfg.get("fusion_experiment", "final_light_bica_gate"))
    experiment_cv = f"{experiment}_{fold}_seed_{seed}"
    run_dir = output_root / "runs" / "fusion" / experiment / fold / f"seed_{seed}"

    _apply_seed(cfg, seed)
    _apply_experiment_identity(
        cfg=cfg,
        experiment_name=experiment_cv,
        run_dir=_as_posix(run_dir),
        output_root=_as_posix(output_root / "runs" / "fusion"),
    )
    _apply_fusion_artifact_paths(cfg, fusion_artifact_dir)
    _append_cv_metadata(cfg, fold, seed, "fusion", source_config_path)

    _assert_no_forbidden_values(
        cfg,
        forbidden_substrings,
        config_name=f"fusion:{fold}:seed_{seed}",
    )

    return cfg, _as_posix(run_dir)


def _quote_cmd(parts: list[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _commands_text(
    text_config: Path,
    speech_config: Path,
    fusion_artifact_config: Path,
    fusion_config: Path,
    text_run_dir: str,
    speech_run_dir: str,
    text_export_dir: Path,
    speech_export_dir: Path,
    seed: int,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated CV fold/seed run plan.",
        "# Text/speech CV runs use fold/seed-specific experiment names.",
        "# Fusion exports/artifacts/runs are isolated under outputs/cv/.",
        "",
        "export PYTHONPATH=.",
        "",
        f"TEXT_RUN_DIR={shlex.quote(text_run_dir)}",
        f"SPEECH_RUN_DIR={shlex.quote(speech_run_dir)}",
        "",
        "# 1) Train text expert",
        'if [ -f "$TEXT_RUN_DIR/resolved_config.yaml" ]; then',
        '  echo "[SKIP] Existing text run: $TEXT_RUN_DIR"',
        "else",
        "  " + _quote_cmd(["python", "scripts/train_text.py", "--config", text_config]),
        "fi",
        "",
        "# 2) Calibrate text expert",
        'if [ -f "$TEXT_RUN_DIR/temperature.json" ]; then',
        '  echo "[SKIP] Existing text calibration: $TEXT_RUN_DIR/temperature.json"',
        "else",
        '  python scripts/calibrate.py --run-dir "$TEXT_RUN_DIR"',
        "fi",
        "",
        "# 3) Export text features",
        _quote_cmd(
            [
                "python",
                "scripts/export_text_features.py",
                "--run-dir",
                text_run_dir,
                "--output-dir",
                text_export_dir,
                "--device",
                "cuda",
            ]
        ),
        "",
        "# 4) Train speech expert",
        'if [ -f "$SPEECH_RUN_DIR/resolved_config.yaml" ]; then',
        '  echo "[SKIP] Existing speech run: $SPEECH_RUN_DIR"',
        "else",
        "  " + _quote_cmd(["python", "scripts/train_speech.py", "--config", speech_config]),
        "fi",
        "",
        "# 5) Calibrate speech expert",
        'if [ -f "$SPEECH_RUN_DIR/temperature.json" ]; then',
        '  echo "[SKIP] Existing speech calibration: $SPEECH_RUN_DIR/temperature.json"',
        "else",
        '  python scripts/calibrate.py --run-dir "$SPEECH_RUN_DIR"',
        "fi",
        "",
        "# 6) Export speech features",
        _quote_cmd(
            [
                "python",
                "scripts/export_speech_features.py",
                "--run-dir",
                speech_run_dir,
                "--output-dir",
                speech_export_dir,
                "--device",
                "cuda",
            ]
        ),
        "",
        "# 7) Prepare fusion artifacts",
        _quote_cmd(
            [
                "python",
                "scripts/prepare_fusion_artifacts.py",
                "--config",
                fusion_artifact_config,
            ]
        ),
        "",
        "# 8) Train/evaluate final fusion",
        _quote_cmd(
            [
                "python",
                "scripts/run_fusion.py",
                "--config",
                fusion_config,
                "--seed",
                seed,
            ]
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def _run_plan(
    fold: str,
    seed: int,
    paths: dict[str, Any],
) -> dict[str, Any]:
    return {
        "fold": fold,
        "seed": int(seed),
        "status": "generated_only",
        "note": (
            "Run commands.sh after reviewing generated configs. "
            "Text/speech CV runs use fold/seed-specific experiment names. "
            "Fusion outputs are under outputs/cv/."
        ),
        "paths": paths,
    }


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_yaml(_resolve(args.config))

    cv_cfg = cfg["cv"]
    source_cfgs = cfg["source_configs"]
    names_cfg = cfg.get("names", {})
    safety_cfg = cfg.get("safety", {})

    forbidden_substrings = list(safety_cfg.get("forbid_value_substrings", []))

    folds_dir = _resolve(cv_cfg["folds_dir"])
    generated_root = _resolve(cv_cfg["generated_config_dir"])
    output_root = _resolve(cv_cfg["output_root"])

    folds = [args.fold] if args.fold else list(cv_cfg.get("folds", []))
    seeds = (
        [int(args.seed)]
        if args.seed is not None
        else [int(x) for x in cv_cfg.get("seeds", [42])]
    )

    if not folds:
        raise ValueError("No folds selected. Provide --fold or cv.folds in config.")

    text_source_path = str(source_cfgs["text"])
    speech_source_path = str(source_cfgs["speech"])
    fusion_source_path = str(source_cfgs["fusion"])

    text_source = _load_source_config(text_source_path)
    speech_source = _load_source_config(speech_source_path)
    fusion_source = _load_source_config(fusion_source_path)

    generated = []

    for fold in folds:
        split_paths = _fold_paths(folds_dir, fold)

        for seed in seeds:
            fold_seed_dir = generated_root / fold / f"seed_{seed}"

            text_export_dir = (
                output_root
                / "fusion_exports"
                / "text_headtail128"
                / fold
                / f"seed_{seed}"
            )
            speech_export_dir = (
                output_root
                / "fusion_exports"
                / "speech_hubert_ce"
                / fold
                / f"seed_{seed}"
            )
            fusion_artifact_dir = (
                output_root
                / "fusion_artifacts"
                / "final_light_bica_gate"
                / fold
                / f"seed_{seed}"
            )

            text_cfg, text_run_dir = _prepare_text_config(
                source_cfg=text_source,
                source_config_path=text_source_path,
                split_paths=split_paths,
                cv_cfg=cv_cfg,
                names_cfg=names_cfg,
                fold=fold,
                seed=seed,
                output_root=output_root,
                forbidden_substrings=forbidden_substrings,
            )

            speech_cfg, speech_run_dir = _prepare_speech_config(
                source_cfg=speech_source,
                source_config_path=speech_source_path,
                split_paths=split_paths,
                cv_cfg=cv_cfg,
                names_cfg=names_cfg,
                fold=fold,
                seed=seed,
                output_root=output_root,
                forbidden_substrings=forbidden_substrings,
            )

            fusion_artifact_cfg = _prepare_fusion_artifact_config(
                source_cfg=fusion_source,
                source_config_path=fusion_source_path,
                fold=fold,
                seed=seed,
                text_run_dir=text_run_dir,
                speech_run_dir=speech_run_dir,
                output_root=output_root,
                text_export_dir=text_export_dir,
                speech_export_dir=speech_export_dir,
                fusion_artifact_dir=fusion_artifact_dir,
                forbidden_substrings=forbidden_substrings,
            )

            fusion_cfg, fusion_run_dir = _prepare_fusion_train_config(
                source_cfg=fusion_source,
                source_config_path=fusion_source_path,
                names_cfg=names_cfg,
                fold=fold,
                seed=seed,
                output_root=output_root,
                fusion_artifact_dir=fusion_artifact_dir,
                forbidden_substrings=forbidden_substrings,
            )

            text_config_path = fold_seed_dir / "text_headtail128.yaml"
            speech_config_path = fold_seed_dir / "speech_hubert_ce.yaml"
            fusion_artifact_config_path = fold_seed_dir / "prepare_fusion_artifacts.yaml"
            fusion_config_path = fold_seed_dir / "fusion_final_light_bica_gate.yaml"
            commands_path = fold_seed_dir / "commands.sh"
            run_plan_path = fold_seed_dir / "run_plan.yaml"

            _safe_write_yaml(text_config_path, text_cfg, overwrite=args.overwrite)
            _safe_write_yaml(speech_config_path, speech_cfg, overwrite=args.overwrite)
            _safe_write_yaml(
                fusion_artifact_config_path,
                fusion_artifact_cfg,
                overwrite=args.overwrite,
            )
            _safe_write_yaml(fusion_config_path, fusion_cfg, overwrite=args.overwrite)

            paths = {
                "generated_dir": _as_posix(fold_seed_dir),
                "text_config": _as_posix(text_config_path),
                "speech_config": _as_posix(speech_config_path),
                "prepare_fusion_artifacts_config": _as_posix(fusion_artifact_config_path),
                "fusion_config": _as_posix(fusion_config_path),
                "commands": _as_posix(commands_path),
                "text_run_dir": text_run_dir,
                "speech_run_dir": speech_run_dir,
                "fusion_run_dir": fusion_run_dir,
                "text_export_dir": _as_posix(text_export_dir),
                "speech_export_dir": _as_posix(speech_export_dir),
                "fusion_artifact_dir": _as_posix(fusion_artifact_dir),
                "split_paths": split_paths,
            }

            _safe_write_yaml(
                run_plan_path,
                _run_plan(fold=fold, seed=seed, paths=paths),
                overwrite=args.overwrite,
            )

            _safe_write_text(
                commands_path,
                _commands_text(
                    text_config=text_config_path,
                    speech_config=speech_config_path,
                    fusion_artifact_config=fusion_artifact_config_path,
                    fusion_config=fusion_config_path,
                    text_run_dir=text_run_dir,
                    speech_run_dir=speech_run_dir,
                    text_export_dir=text_export_dir,
                    speech_export_dir=speech_export_dir,
                    seed=seed,
                ),
                overwrite=args.overwrite,
            )
            commands_path.chmod(0o755)

            generated.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "run_plan": _as_posix(run_plan_path),
                    "commands": _as_posix(commands_path),
                }
            )

    summary_path = generated_root / "generation_summary.json"
    _safe_write_text(
        summary_path,
        json.dumps(generated, ensure_ascii=False, indent=2),
        overwrite=True,
    )

    print(f"[OK] Generated CV fold configs under: {generated_root}")
    print(json.dumps(generated, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())