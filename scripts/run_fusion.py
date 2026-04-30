from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.checkpoint import load_checkpoint
from src.common.registry import MODEL_REGISTRY
from src.evaluation.classification_report import (
    build_classification_report,
    save_classification_report,
)
from src.evaluation.confusion_matrix import (
    build_confusion_matrix_artifact,
    save_confusion_matrix,
)
from src.evaluation.metrics import compute_classification_metrics
from src.fusion.features import build_modality_priors_from_valid_artifact
from src.fusion.io import load_torch_artifact, validate_fusion_ready_artifact
from src.fusion.model import FusionClassifier  # noqa: F401
from src.fusion.trainer import FusionTrainer, build_fusion_dataloaders, save_epoch_predictions
from src.utils.config import deep_update, load_yaml
from src.utils.io import write_json, write_yaml
from src.utils.paths import resolve_project_path
from src.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate fusion model.")
    parser.add_argument("--config", type=Path, required=True, help="Fusion experiment config.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/runs/fusion"),
        help="Base directory for fusion runs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, auto")
    parser.add_argument("--resume-from", type=Path, default=None, help="Optional checkpoint path.")
    return parser


def resolve_experiment_config(config_path: Path) -> dict[str, Any]:
    exp_cfg = load_yaml(resolve_project_path(config_path, PROJECT_ROOT))
    defaults = exp_cfg.get("defaults", [])
    merged: dict[str, Any] = {}

    for entry in defaults:
        if not isinstance(entry, str):
            raise TypeError("Fusion config `defaults` entries must be string paths.")
        cfg_part = load_yaml(resolve_project_path(entry, PROJECT_ROOT))
        merged = deep_update(merged, cfg_part)

    exp_without_defaults = dict(exp_cfg)
    exp_without_defaults.pop("defaults", None)
    merged = deep_update(merged, exp_without_defaults)
    merged.setdefault("project", {})
    merged["project"].setdefault("root", str(PROJECT_ROOT))
    return merged


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(config)
    cfg["runtime"] = dict(cfg.get("runtime", {}))

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    return cfg


def infer_model_dims(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config)
    cfg["model"] = dict(cfg["model"])
    project_root = cfg.get("project", {}).get("root", ".")
    train_artifact = load_torch_artifact(
        resolve_project_path(cfg["dataset"]["fusion_artifacts"]["train"], project_root)
    )
    validate_fusion_ready_artifact(train_artifact)

    cfg["model"]["text_token_dim"] = int(train_artifact["text_tokens"].shape[-1])
    cfg["model"]["speech_token_dim"] = int(train_artifact["speech_tokens"].shape[-1])
    cfg["model"]["text_embedding_dim"] = int(train_artifact["text_embedding"].shape[-1])
    cfg["model"]["speech_embedding_dim"] = int(train_artifact["speech_embedding"].shape[-1])
    cfg["model"]["reliability_dim"] = int(train_artifact["reliability"].shape[-1])
    return cfg


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_name = str(config["model"]["name"])
    model_cls = MODEL_REGISTRY.get(model_name)
    return model_cls(config)


def build_run_dir(config: dict[str, Any], output_root: Path, seed: int) -> Path:
    experiment_name = config.get("experiment", {}).get("name") or config["model"]["name"]
    run_dir = output_root / experiment_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_metrics_json(
    output_path: Path,
    fit_result: dict[str, Any],
    valid_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    branch_metrics: dict[str, Any],
) -> None:
    payload = {
        "fit": fit_result,
        "valid": valid_metrics,
        "test": test_metrics,
        "branches": branch_metrics,
    }
    write_json(payload, output_path)


def _extract_branch_metrics(result, label_names: list[str]) -> dict[str, Any]:
    y_true = result.predictions.labels
    metadata = result.predictions.metadata

    branch_map = {
        "text_only": metadata.get("text_pred_id", []),
        "speech_only": metadata.get("speech_pred_id", []),
        "interaction_only": metadata.get("interaction_pred_id", []),
        "fused_final": result.predictions.preds,
    }

    metrics: dict[str, Any] = {}
    for branch_name, preds in branch_map.items():
        if preds and len(preds) == len(y_true):
            metrics[branch_name] = compute_classification_metrics(
                y_true=y_true,
                y_pred=preds,
                class_names=label_names,
            )
    return metrics


def main() -> int:
    args = build_parser().parse_args()

    config = resolve_experiment_config(args.config)
    config = apply_cli_overrides(config, args)
    config = infer_model_dims(config)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    run_dir = build_run_dir(config, args.output_root, seed)
    train_loader, valid_loader, test_loader = build_fusion_dataloaders(config)

    project_root = config.get("project", {}).get("root", ".")
    valid_artifact = load_torch_artifact(
        resolve_project_path(config["dataset"]["fusion_artifacts"]["valid"], project_root)
    )
    validate_fusion_ready_artifact(valid_artifact)
    prior_text, prior_speech = build_modality_priors_from_valid_artifact(valid_artifact)

    write_yaml(config, run_dir / "resolved_config.yaml")
    write_json(
        {
            "prior_text": prior_text.tolist(),
            "prior_speech": prior_speech.tolist(),
        },
        run_dir / "dominance_priors.json",
    )

    model = build_model(config)
    if args.resume_from is not None:
        ckpt = load_checkpoint(args.resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    label_names = config["label_space"]["labels"]

    trainer = FusionTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        class_names=label_names,
        run_dir=run_dir,
        prior_text=prior_text,
        prior_speech=prior_speech,
    )

    fit_result = trainer.fit()

    best_ckpt_path = trainer.checkpoint_manager.best_path
    if best_ckpt_path.exists():
        ckpt = load_checkpoint(best_ckpt_path, map_location="cpu")
        trainer.model.load_state_dict(ckpt["model_state_dict"])

    valid_result = trainer.evaluate(valid_loader, split="valid")
    test_result = trainer.evaluate(test_loader, split="test")

    save_epoch_predictions(valid_result, run_dir / "valid_predictions.csv", label_names=label_names)
    save_epoch_predictions(test_result, run_dir / "test_predictions.csv", label_names=label_names)

    branch_metrics = {
        "valid": _extract_branch_metrics(valid_result, label_names),
        "test": _extract_branch_metrics(test_result, label_names),
    }
    write_json(branch_metrics, run_dir / "branch_metrics.json")

    if config.get("output", {}).get("save_confusion_matrix", True):
        cm_artifact = build_confusion_matrix_artifact(
            y_true=test_result.predictions.labels,
            y_pred=test_result.predictions.preds,
            labels=list(range(len(label_names))),
            class_names=label_names,
        )
        save_confusion_matrix(cm_artifact, run_dir / "confusion_matrix.json")

    if config.get("output", {}).get("save_classification_report", True):
        report = build_classification_report(
            y_true=test_result.predictions.labels,
            y_pred=test_result.predictions.preds,
            target_names=label_names,
        )
        save_classification_report(report, run_dir / "classification_report.json")

    save_metrics_json(
        run_dir / "metrics.json",
        fit_result=fit_result,
        valid_metrics=valid_result.metrics,
        test_metrics=test_result.metrics,
        branch_metrics=branch_metrics,
    )

    print(f"[OK] Finished fusion run: {run_dir}")
    print(json.dumps(test_result.metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())