# scripts/generate_meld_eval_configs.py
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

import yaml


FORBIDDEN_FRAGMENTS = [
    "outputs/cv",
    "/outputs/cv/",
    "outputs/runs/text_length_aware_tuning",
    "/outputs/runs/text_length_aware_tuning/",
    "outputs/runs/fusion_text_headtail128_export_fixed",
    "/outputs/runs/fusion_text_headtail128_export_fixed/",
    "outputs/runs/text/",
    "/outputs/runs/text/",
    "outputs/runs/speech/",
    "/outputs/runs/speech/",
    "outputs/runs/fusion/",
    "/outputs/runs/fusion/",
    "fold_00_seed_42/seed_42",
]


def _resolve(path: str | Path, project_root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return project_root / path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML template not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _set_nested(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    current: dict[str, Any] = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[parts[-1]] = value


def _set_key_everywhere(obj: Any, key: str, value: Any) -> None:
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k == key:
                obj[k] = value
            else:
                _set_key_everywhere(obj[k], key, value)
    elif isinstance(obj, list):
        for item in obj:
            _set_key_everywhere(item, key, value)


def _replace_strings(obj: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(obj, dict):
        return {k: _replace_strings(v, replacements) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_strings(item, replacements) for item in obj]
    if isinstance(obj, str):
        out = obj
        for old, new in replacements:
            out = out.replace(old, new)
        return out
    return obj


def _collect_forbidden_strings(
    obj: Any,
    *,
    prefix: str = "$",
) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            hits.extend(_collect_forbidden_strings(value, prefix=f"{prefix}.{key}"))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            hits.extend(_collect_forbidden_strings(value, prefix=f"{prefix}[{idx}]"))
    elif isinstance(obj, str):
        for frag in FORBIDDEN_FRAGMENTS:
            if frag in obj:
                hits.append({"path": prefix, "fragment": frag, "value": obj})

    return hits


def _build_meld_path_replacements(
    *,
    project_root: Path,
    text_run_root: str,
    speech_run_root: str,
    fusion_run_root: str,
    text_run_dir: str,
    speech_run_dir: str,
    text_export_dir: str,
    speech_export_dir: str,
    artifact_dir: str,
    fusion_run_dir: str,
    text_calibration_path: str,
    speech_calibration_path: str,
) -> list[tuple[str, str]]:
    root = str(project_root)

    old_to_new = {
        f"{root}/outputs/runs/text": text_run_root,
        "outputs/runs/text": text_run_root,
        f"{root}/outputs/runs/speech": speech_run_root,
        "outputs/runs/speech": speech_run_root,
        f"{root}/outputs/runs/fusion": fusion_run_root,
        "outputs/runs/fusion": fusion_run_root,

        f"{root}/outputs/cv/runs/text/text_phobert_base_ce_headtail128_fold_00_seed_42/seed_42": text_run_dir,
        "outputs/cv/runs/text/text_phobert_base_ce_headtail128_fold_00_seed_42/seed_42": text_run_dir,
        f"{root}/outputs/cv/runs/speech/speech_hubert_base_attn_freeze_ce_fold_00_seed_42/seed_42": speech_run_dir,
        "outputs/cv/runs/speech/speech_hubert_base_attn_freeze_ce_fold_00_seed_42/seed_42": speech_run_dir,

        f"{root}/outputs/meld/runs/text/text_phobert_base_ce_headtail128_fold_00_seed_42/seed_42": text_run_dir,
        "outputs/meld/runs/text/text_phobert_base_ce_headtail128_fold_00_seed_42/seed_42": text_run_dir,
        f"{root}/outputs/meld/runs/speech/speech_hubert_base_attn_freeze_ce_fold_00_seed_42/seed_42": speech_run_dir,
        "outputs/meld/runs/speech/speech_hubert_base_attn_freeze_ce_fold_00_seed_42/seed_42": speech_run_dir,

        f"{root}/outputs/cv/fusion_exports/text_headtail128/fold_00/seed_42": text_export_dir,
        "outputs/cv/fusion_exports/text_headtail128/fold_00/seed_42": text_export_dir,
        f"{root}/outputs/cv/fusion_exports/speech_hubert_ce/fold_00/seed_42": speech_export_dir,
        "outputs/cv/fusion_exports/speech_hubert_ce/fold_00/seed_42": speech_export_dir,

        f"{root}/outputs/cv/fusion_artifacts/final_light_bica_gate/fold_00/seed_42": artifact_dir,
        "outputs/cv/fusion_artifacts/final_light_bica_gate/fold_00/seed_42": artifact_dir,

        f"{root}/outputs/cv/runs/fusion/fusion_light_bica_gate_text_headtail128_cal_gateprob_v1/fold_00/seed_42": fusion_run_dir,
        "outputs/cv/runs/fusion/fusion_light_bica_gate_text_headtail128_cal_gateprob_v1/fold_00/seed_42": fusion_run_dir,

        f"{root}/outputs/runs/text_length_aware_tuning/text_phobert_base_ce_headtail128/seed_42/temperature.json": text_calibration_path,
        "outputs/runs/text_length_aware_tuning/text_phobert_base_ce_headtail128/seed_42/temperature.json": text_calibration_path,
        f"{root}/outputs/runs/text_length_aware_tuning/text_phobert_base_ce_headtail128/seed_42": text_run_dir,
        "outputs/runs/text_length_aware_tuning/text_phobert_base_ce_headtail128/seed_42": text_run_dir,

        f"{root}/outputs/runs/speech/speech_hubert_base_attn_freeze_ce/seed_42/temperature.json": speech_calibration_path,
        "outputs/runs/speech/speech_hubert_base_attn_freeze_ce/seed_42/temperature.json": speech_calibration_path,
        f"{root}/outputs/runs/speech/speech_hubert_base_attn_freeze_ce/seed_42": speech_run_dir,
        "outputs/runs/speech/speech_hubert_base_attn_freeze_ce/seed_42": speech_run_dir,

        f"{root}/outputs/runs/fusion_text_headtail128_export_fixed/fusion_light_bica_gate_text_headtail128_cal_gateprob_v1/seed_42": fusion_run_dir,
        "outputs/runs/fusion_text_headtail128_export_fixed/fusion_light_bica_gate_text_headtail128_cal_gateprob_v1/seed_42": fusion_run_dir,
    }

    return sorted(old_to_new.items(), key=lambda item: len(item[0]), reverse=True)


def _sanitize_generated_config(
    cfg: dict[str, Any],
    *,
    replacements: list[tuple[str, str]],
    config_name: str,
) -> dict[str, Any]:
    sanitized = _replace_strings(cfg, replacements)
    hits = _collect_forbidden_strings(sanitized)

    if hits:
        raise ValueError(
            f"Forbidden stale VNEMOS/CV paths remain in generated MELD config `{config_name}`. "
            f"Examples: {hits[:20]}"
        )

    return sanitized


def _patch_seed(cfg: dict[str, Any], seed: int) -> None:
    cfg["seed"] = seed
    for dotted in [
        "training.seed",
        "trainer.seed",
        "experiment.seed",
        "run.seed",
    ]:
        _set_nested(cfg, dotted, seed)
    _set_key_everywhere(cfg, "random_seed", seed)


def _patch_dataset_csvs(
    cfg: dict[str, Any],
    *,
    train_csv: str,
    valid_csv: str,
    test_csv: str,
) -> None:
    _set_nested(cfg, "dataset.train_csv", train_csv)
    _set_nested(cfg, "dataset.valid_csv", valid_csv)
    _set_nested(cfg, "dataset.test_csv", test_csv)

    _set_key_everywhere(cfg, "train_csv", train_csv)
    _set_key_everywhere(cfg, "valid_csv", valid_csv)
    _set_key_everywhere(cfg, "test_csv", test_csv)


def _patch_experiment_name(cfg: dict[str, Any], run_name: str) -> None:
    cfg["experiment_name"] = run_name
    _set_nested(cfg, "experiment.name", run_name)
    _set_nested(cfg, "run.name", run_name)


def _patch_train_output_paths(
    cfg: dict[str, Any],
    *,
    run_name: str,
    run_dir: str,
    output_root: str,
) -> None:
    _patch_experiment_name(cfg, run_name)

    cfg["output_dir"] = run_dir
    cfg["run_dir"] = run_dir
    cfg["output_root"] = output_root

    for dotted in [
        "output.root",
        "outputs.root",
        "training.root",
        "trainer.root",
        "run.root",
        "experiment.root",
        "training.output_root",
        "trainer.output_root",
        "run.output_root",
        "experiment.output_root",
    ]:
        _set_nested(cfg, dotted, output_root)

    for dotted in [
        "training.output_dir",
        "trainer.output_dir",
        "run.output_dir",
        "run.run_dir",
        "experiment.output_dir",
        "experiment.run_dir",
    ]:
        _set_nested(cfg, dotted, run_dir)


def _build_export_metadata_config(
    *,
    role: str,
    run_dir: str,
    export_dir: str,
    train_csv: str,
    valid_csv: str,
    test_csv: str,
) -> dict[str, Any]:
    return {
        "role": role,
        "note": "Metadata-only config. The command uses --run-dir and --output-dir directly.",
        "run_dir": run_dir,
        "export_dir": export_dir,
        "output_dir": export_dir,
        "dataset": {
            "train_csv": train_csv,
            "valid_csv": valid_csv,
            "test_csv": test_csv,
        },
    }


def _build_prepare_fusion_config(
    *,
    text_export_dir: str,
    speech_export_dir: str,
    text_calibration_path: str,
    speech_calibration_path: str,
    artifact_dir: str,
) -> dict[str, Any]:
    artifacts = {
        "train": f"{artifact_dir}/train.pt",
        "valid": f"{artifact_dir}/valid.pt",
        "test": f"{artifact_dir}/test.pt",
        "train_artifact": f"{artifact_dir}/train.pt",
        "valid_artifact": f"{artifact_dir}/valid.pt",
        "test_artifact": f"{artifact_dir}/test.pt",
    }

    calibration = {
        "required": True,
        "text": text_calibration_path,
        "speech": speech_calibration_path,
    }

    return {
        "role": "prepare_fusion_artifacts",
        "output_dir": artifact_dir,
        "artifact_dir": artifact_dir,
        "fusion_artifact_dir": artifact_dir,
        "dataset": {
            "text_export_dir": text_export_dir,
            "speech_export_dir": speech_export_dir,
            "output_dir": artifact_dir,
            "calibration": calibration,
            "fusion_artifacts": {
                "train": f"{artifact_dir}/train.pt",
                "valid": f"{artifact_dir}/valid.pt",
                "test": f"{artifact_dir}/test.pt",
            },
            **artifacts,
            "train": f"{artifact_dir}/train.pt",
            "valid": f"{artifact_dir}/valid.pt",
            "test": f"{artifact_dir}/test.pt",
        },
        "calibration": calibration,
        "fusion_artifacts": {
            "output_dir": artifact_dir,
            "train": f"{artifact_dir}/train.pt",
            "valid": f"{artifact_dir}/valid.pt",
            "test": f"{artifact_dir}/test.pt",
            **artifacts,
        },
    }


def _patch_fusion_train_config(
    cfg: dict[str, Any],
    *,
    run_name: str,
    artifact_dir: str,
    run_dir: str,
    output_root: str,
    seed: int,
    text_calibration_path: str,
    speech_calibration_path: str,
) -> None:
    _patch_seed(cfg, seed)
    _patch_experiment_name(cfg, run_name)

    cfg["output_dir"] = run_dir
    cfg["run_dir"] = run_dir
    cfg["output_root"] = output_root

    for dotted in [
        "output.root",
        "outputs.root",
        "training.root",
        "trainer.root",
        "run.root",
        "experiment.root",
        "training.output_root",
        "trainer.output_root",
        "run.output_root",
        "experiment.output_root",
    ]:
        _set_nested(cfg, dotted, output_root)

    for dotted in [
        "training.output_dir",
        "trainer.output_dir",
        "run.output_dir",
        "run.run_dir",
        "experiment.output_dir",
        "experiment.run_dir",
    ]:
        _set_nested(cfg, dotted, run_dir)

    _set_nested(cfg, "dataset.train_artifact", f"{artifact_dir}/train.pt")
    _set_nested(cfg, "dataset.valid_artifact", f"{artifact_dir}/valid.pt")
    _set_nested(cfg, "dataset.test_artifact", f"{artifact_dir}/test.pt")
    _set_nested(cfg, "dataset.train", f"{artifact_dir}/train.pt")
    _set_nested(cfg, "dataset.valid", f"{artifact_dir}/valid.pt")
    _set_nested(cfg, "dataset.test", f"{artifact_dir}/test.pt")
    _set_nested(cfg, "dataset.fusion_artifacts.train", f"{artifact_dir}/train.pt")
    _set_nested(cfg, "dataset.fusion_artifacts.valid", f"{artifact_dir}/valid.pt")
    _set_nested(cfg, "dataset.fusion_artifacts.test", f"{artifact_dir}/test.pt")

    _set_nested(cfg, "fusion_artifacts.train", f"{artifact_dir}/train.pt")
    _set_nested(cfg, "fusion_artifacts.valid", f"{artifact_dir}/valid.pt")
    _set_nested(cfg, "fusion_artifacts.test", f"{artifact_dir}/test.pt")
    _set_nested(cfg, "fusion_artifacts.train_artifact", f"{artifact_dir}/train.pt")
    _set_nested(cfg, "fusion_artifacts.valid_artifact", f"{artifact_dir}/valid.pt")
    _set_nested(cfg, "fusion_artifacts.test_artifact", f"{artifact_dir}/test.pt")

    _set_nested(cfg, "calibration.text", text_calibration_path)
    _set_nested(cfg, "calibration.speech", speech_calibration_path)
    _set_nested(cfg, "dataset.calibration.text", text_calibration_path)
    _set_nested(cfg, "dataset.calibration.speech", speech_calibration_path)


def _q(path: str | Path) -> str:
    return shlex.quote(str(path))


def _write_commands(
    path: Path,
    *,
    python_cmd: str,
    seed: int,
    text_cfg: Path,
    speech_cfg: Path,
    prepare_cfg: Path,
    fusion_cfg: Path,
    text_run_root: str,
    speech_run_root: str,
    text_run_dir: str,
    speech_run_dir: str,
    text_export_dir: str,
    speech_export_dir: str,
    artifact_dir: str,
    fusion_run_dir: str,
) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "export PYTHONPATH=${PYTHONPATH:-.}",
        "",
        f"TEXT_RUN_ROOT={shlex.quote(text_run_root)}",
        f"SPEECH_RUN_ROOT={shlex.quote(speech_run_root)}",
        f"TEXT_RUN_DIR={shlex.quote(text_run_dir)}",
        f"SPEECH_RUN_DIR={shlex.quote(speech_run_dir)}",
        f"TEXT_EXPORT_DIR={shlex.quote(text_export_dir)}",
        f"SPEECH_EXPORT_DIR={shlex.quote(speech_export_dir)}",
        f"FUSION_ARTIFACT_DIR={shlex.quote(artifact_dir)}",
        f"FUSION_RUN_DIR={shlex.quote(fusion_run_dir)}",
        "",
        'echo "[MELD] Train text expert"',
        'if [ -f "$TEXT_RUN_DIR/metrics.json" ]; then',
        '  echo "[SKIP] Existing text run: $TEXT_RUN_DIR"',
        "else",
        f"  {python_cmd} scripts/train_text.py --config {_q(text_cfg)} --seed {seed} --output-root \"$TEXT_RUN_ROOT\"",
        "fi",
        "",
        'echo "[MELD] Calibrate text expert"',
        'if [ ! -d "$TEXT_RUN_DIR" ]; then',
        '  echo "[ERROR] Expected text run dir not found: $TEXT_RUN_DIR"',
        '  echo "[ERROR] train_text.py likely wrote to the wrong output-root."',
        "  exit 1",
        "fi",
        'if [ -f "$TEXT_RUN_DIR/temperature.json" ]; then',
        '  echo "[SKIP] Existing text calibration: $TEXT_RUN_DIR/temperature.json"',
        "else",
        f"  {python_cmd} scripts/calibrate.py --run-dir \"$TEXT_RUN_DIR\"",
        "fi",
        "",
        'echo "[MELD] Export text features"',
        'if [ -f "$TEXT_EXPORT_DIR/train.pt" ] && [ -f "$TEXT_EXPORT_DIR/valid.pt" ] && [ -f "$TEXT_EXPORT_DIR/test.pt" ]; then',
        '  echo "[SKIP] Existing text exports: $TEXT_EXPORT_DIR"',
        "else",
        f"  {python_cmd} scripts/export_text_features.py --run-dir \"$TEXT_RUN_DIR\" --output-dir \"$TEXT_EXPORT_DIR\"",
        "fi",
        "",
        'echo "[MELD] Train speech expert"',
        'if [ -f "$SPEECH_RUN_DIR/metrics.json" ]; then',
        '  echo "[SKIP] Existing speech run: $SPEECH_RUN_DIR"',
        "else",
        f"  {python_cmd} scripts/train_speech.py --config {_q(speech_cfg)} --seed {seed} --output-root \"$SPEECH_RUN_ROOT\"",
        "fi",
        "",
        'echo "[MELD] Calibrate speech expert"',
        'if [ ! -d "$SPEECH_RUN_DIR" ]; then',
        '  echo "[ERROR] Expected speech run dir not found: $SPEECH_RUN_DIR"',
        '  echo "[ERROR] train_speech.py likely wrote to the wrong output-root."',
        "  exit 1",
        "fi",
        'if [ -f "$SPEECH_RUN_DIR/temperature.json" ]; then',
        '  echo "[SKIP] Existing speech calibration: $SPEECH_RUN_DIR/temperature.json"',
        "else",
        f"  {python_cmd} scripts/calibrate.py --run-dir \"$SPEECH_RUN_DIR\"",
        "fi",
        "",
        'echo "[MELD] Export speech features"',
        'if [ -f "$SPEECH_EXPORT_DIR/train.pt" ] && [ -f "$SPEECH_EXPORT_DIR/valid.pt" ] && [ -f "$SPEECH_EXPORT_DIR/test.pt" ]; then',
        '  echo "[SKIP] Existing speech exports: $SPEECH_EXPORT_DIR"',
        "else",
        f"  {python_cmd} scripts/export_speech_features.py --run-dir \"$SPEECH_RUN_DIR\" --output-dir \"$SPEECH_EXPORT_DIR\"",
        "fi",
        "",
        'echo "[MELD] Prepare fusion artifacts"',
        'if [ -f "$FUSION_ARTIFACT_DIR/train.pt" ] && [ -f "$FUSION_ARTIFACT_DIR/valid.pt" ] && [ -f "$FUSION_ARTIFACT_DIR/test.pt" ]; then',
        '  echo "[SKIP] Existing fusion artifacts: $FUSION_ARTIFACT_DIR"',
        "else",
        f"  {python_cmd} scripts/prepare_fusion_artifacts.py --config {_q(prepare_cfg)}",
        "fi",
        "",
        'echo "[MELD] Train fusion model"',
        'if [ -f "$FUSION_RUN_DIR/metrics.json" ]; then',
        '  echo "[SKIP] Existing fusion run: $FUSION_RUN_DIR"',
        "else",
        f"  {python_cmd} scripts/run_fusion.py --config {_q(fusion_cfg)}",
        "fi",
        "",
        'echo "[OK] MELD framework evaluation finished."',
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate add-only MELD evaluation configs for the final Light-BiCA-Gate framework."
    )
    parser.add_argument("--config", default="configs/meld/meld_light_bica_gate_eval.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    master_path = _resolve(args.config, project_root)
    master = _load_yaml(master_path)

    seeds = [int(args.seed)] if args.seed is not None else [int(s) for s in master.get("seeds", [42])]

    dataset = dict(master.get("dataset", {}))
    train_csv = str(_resolve(dataset["train_csv"], project_root))
    valid_csv = str(_resolve(dataset["valid_csv"], project_root))
    test_csv = str(_resolve(dataset["test_csv"], project_root))

    base = dict(master.get("base_configs", {}))
    outputs = dict(master.get("outputs", {}))
    names = dict(master.get("names", {}))
    commands_cfg = dict(master.get("commands", {}))

    generated_root = _resolve(
        master.get("generated_config_root", "configs/meld/generated/meld_light_bica_gate"),
        project_root,
    )
    python_cmd = str(commands_cfg.get("python", "python"))

    generated: list[dict[str, Any]] = []

    for seed in seeds:
        seed_dir = generated_root / f"seed_{seed}"
        if seed_dir.exists() and not args.overwrite:
            raise FileExistsError(f"Generated seed directory already exists: {seed_dir}. Use --overwrite.")

        text_run_name = str(names.get("text_run_name", "text_phobert_base_ce_headtail128"))
        speech_run_name = str(names.get("speech_run_name", "speech_hubert_base_attn_freeze_ce"))
        fusion_run_name = str(names.get("fusion_run_name", "fusion_light_bica_gate_text_headtail128_cal_gateprob_v1"))

        text_run_root = _resolve(outputs.get("text_run_root", "outputs/meld/runs/text"), project_root)
        speech_run_root = _resolve(outputs.get("speech_run_root", "outputs/meld/runs/speech"), project_root)
        text_export_root = _resolve(outputs.get("text_export_root", "outputs/meld/fusion_exports/text_headtail128"), project_root)
        speech_export_root = _resolve(outputs.get("speech_export_root", "outputs/meld/fusion_exports/speech_hubert"), project_root)
        artifact_root = _resolve(outputs.get("fusion_artifact_root", "outputs/meld/fusion_artifacts/light_bica_gate"), project_root)
        fusion_run_root = _resolve(outputs.get("fusion_run_root", "outputs/meld/runs/fusion"), project_root)

        text_run_dir = str(text_run_root / text_run_name / f"seed_{seed}")
        speech_run_dir = str(speech_run_root / speech_run_name / f"seed_{seed}")
        text_export_dir = str(text_export_root / f"seed_{seed}")
        speech_export_dir = str(speech_export_root / f"seed_{seed}")
        artifact_dir = str(artifact_root / f"seed_{seed}")
        fusion_run_dir = str(fusion_run_root / fusion_run_name / f"seed_{seed}")

        text_calibration_path = f"{text_run_dir}/temperature.json"
        speech_calibration_path = f"{speech_run_dir}/temperature.json"

        text_cfg = _load_yaml(_resolve(base["text_train"], project_root))
        _patch_seed(text_cfg, seed)
        _patch_dataset_csvs(text_cfg, train_csv=train_csv, valid_csv=valid_csv, test_csv=test_csv)
        _patch_train_output_paths(
            text_cfg,
            run_name=text_run_name,
            run_dir=text_run_dir,
            output_root=str(text_run_root),
        )

        speech_cfg = _load_yaml(_resolve(base["speech_train"], project_root))
        _patch_seed(speech_cfg, seed)
        _patch_dataset_csvs(speech_cfg, train_csv=train_csv, valid_csv=valid_csv, test_csv=test_csv)
        _patch_train_output_paths(
            speech_cfg,
            run_name=speech_run_name,
            run_dir=speech_run_dir,
            output_root=str(speech_run_root),
        )

        text_export_cfg = _build_export_metadata_config(
            role="export_text_features",
            run_dir=text_run_dir,
            export_dir=text_export_dir,
            train_csv=train_csv,
            valid_csv=valid_csv,
            test_csv=test_csv,
        )

        speech_export_cfg = _build_export_metadata_config(
            role="export_speech_features",
            run_dir=speech_run_dir,
            export_dir=speech_export_dir,
            train_csv=train_csv,
            valid_csv=valid_csv,
            test_csv=test_csv,
        )

        prepare_cfg = _build_prepare_fusion_config(
            text_export_dir=text_export_dir,
            speech_export_dir=speech_export_dir,
            text_calibration_path=text_calibration_path,
            speech_calibration_path=speech_calibration_path,
            artifact_dir=artifact_dir,
        )

        fusion_cfg = _load_yaml(_resolve(base["fusion_train"], project_root))
        _patch_fusion_train_config(
            fusion_cfg,
            run_name=fusion_run_name,
            artifact_dir=artifact_dir,
            run_dir=fusion_run_dir,
            output_root=str(fusion_run_root),
            seed=seed,
            text_calibration_path=text_calibration_path,
            speech_calibration_path=speech_calibration_path,
        )

        replacements = _build_meld_path_replacements(
            project_root=project_root,
            text_run_root=str(text_run_root),
            speech_run_root=str(speech_run_root),
            fusion_run_root=str(fusion_run_root),
            text_run_dir=text_run_dir,
            speech_run_dir=speech_run_dir,
            text_export_dir=text_export_dir,
            speech_export_dir=speech_export_dir,
            artifact_dir=artifact_dir,
            fusion_run_dir=fusion_run_dir,
            text_calibration_path=text_calibration_path,
            speech_calibration_path=speech_calibration_path,
        )

        text_cfg = _sanitize_generated_config(text_cfg, replacements=replacements, config_name="text_train")
        speech_cfg = _sanitize_generated_config(speech_cfg, replacements=replacements, config_name="speech_train")
        text_export_cfg = _sanitize_generated_config(text_export_cfg, replacements=replacements, config_name="text_export")
        speech_export_cfg = _sanitize_generated_config(speech_export_cfg, replacements=replacements, config_name="speech_export")
        prepare_cfg = _sanitize_generated_config(prepare_cfg, replacements=replacements, config_name="prepare_fusion_artifacts")
        fusion_cfg = _sanitize_generated_config(fusion_cfg, replacements=replacements, config_name="fusion_train")

        text_cfg_path = seed_dir / "text_phobert_headtail128.yaml"
        speech_cfg_path = seed_dir / "speech_hubert.yaml"
        text_export_cfg_path = seed_dir / "export_text_headtail128.yaml"
        speech_export_cfg_path = seed_dir / "export_speech_hubert.yaml"
        prepare_cfg_path = seed_dir / "prepare_fusion_artifacts.yaml"
        fusion_cfg_path = seed_dir / "fusion_light_bica_gate.yaml"
        commands_path = seed_dir / "commands.sh"
        run_plan_path = seed_dir / "run_plan.yaml"

        _write_yaml(text_cfg_path, text_cfg)
        _write_yaml(speech_cfg_path, speech_cfg)
        _write_yaml(text_export_cfg_path, text_export_cfg)
        _write_yaml(speech_export_cfg_path, speech_export_cfg)
        _write_yaml(prepare_cfg_path, prepare_cfg)
        _write_yaml(fusion_cfg_path, fusion_cfg)

        run_plan = {
            "setting": master.get("setting_name", "MELD-ViText-OriginalAudio"),
            "seed": seed,
            "dataset": {
                "train_csv": train_csv,
                "valid_csv": valid_csv,
                "test_csv": test_csv,
            },
            "outputs": {
                "text_run_dir": text_run_dir,
                "speech_run_dir": speech_run_dir,
                "text_export_dir": text_export_dir,
                "speech_export_dir": speech_export_dir,
                "fusion_artifact_dir": artifact_dir,
                "fusion_run_dir": fusion_run_dir,
            },
            "configs": {
                "text": str(text_cfg_path),
                "speech": str(speech_cfg_path),
                "text_export": str(text_export_cfg_path),
                "speech_export": str(speech_export_cfg_path),
                "prepare_fusion_artifacts": str(prepare_cfg_path),
                "fusion": str(fusion_cfg_path),
            },
            "commands": str(commands_path),
            "note": "All MELD outputs are isolated under outputs/meld and do not overwrite VNEMOS baseline outputs.",
        }
        _write_yaml(run_plan_path, run_plan)

        _write_commands(
            commands_path,
            python_cmd=python_cmd,
            seed=seed,
            text_cfg=text_cfg_path,
            speech_cfg=speech_cfg_path,
            prepare_cfg=prepare_cfg_path,
            fusion_cfg=fusion_cfg_path,
            text_run_root=str(text_run_root),
            speech_run_root=str(speech_run_root),
            text_run_dir=text_run_dir,
            speech_run_dir=speech_run_dir,
            text_export_dir=text_export_dir,
            speech_export_dir=speech_export_dir,
            artifact_dir=artifact_dir,
            fusion_run_dir=fusion_run_dir,
        )

        generated.append(
            {
                "seed": seed,
                "run_plan": str(run_plan_path),
                "commands": str(commands_path),
                "text_run_dir": text_run_dir,
                "speech_run_dir": speech_run_dir,
                "fusion_run_dir": fusion_run_dir,
            }
        )

    print(f"[OK] Generated MELD evaluation configs under: {generated_root}")
    print(json.dumps(generated, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())