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
from src.speech.error_analysis import save_basic_error_analysis
from src.speech.models.hubert import HuBERTClassifier  # noqa: F401
from src.speech.processors import build_speech_processor
from src.speech.trainer import (
    SpeechTrainer,
    build_speech_dataloaders,
    epoch_result_to_dataframe,
    save_epoch_predictions,
)
from src.utils.config import deep_update, load_yaml
from src.utils.io import write_json, write_yaml
from src.utils.paths import resolve_project_path
from src.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a speech emotion recognition model."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a speech experiment config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/runs/speech"),
        help="Base directory for speech run outputs.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override: cpu, cuda, auto.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode using cpu-like overrides.",
    )
    return parser


def resolve_experiment_config(config_path: Path) -> dict[str, Any]:
    exp_cfg = load_yaml(resolve_project_path(config_path, PROJECT_ROOT))
    defaults = exp_cfg.get("defaults", [])
    merged: dict[str, Any] = {}

    for entry in defaults:
        if not isinstance(entry, str):
            raise TypeError("Speech config `defaults` entries must be string paths.")
        cfg_part = load_yaml(resolve_project_path(entry, PROJECT_ROOT))
        merged = deep_update(merged, cfg_part)

    exp_without_defaults = dict(exp_cfg)
    exp_without_defaults.pop("defaults", None)
    merged = deep_update(merged, exp_without_defaults)
    return merged


def apply_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict(config)
    cfg["runtime"] = dict(cfg.get("runtime", {}))
    cfg["train"] = dict(cfg.get("train", {}))

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device is not None:
        cfg["runtime"]["device"] = args.device

    if args.debug:
        cfg["runtime"]["device"] = "cpu"
        cfg["runtime"]["mixed_precision"] = "none"
        cfg["train"]["epochs"] = 1
        cfg["train"]["batch_size"] = min(int(cfg["train"].get("batch_size", 8)), 4)
        cfg.setdefault("early_stopping", {})
        cfg["early_stopping"]["enabled"] = False

    return cfg


def build_run_dir(config: dict[str, Any], output_root: Path, seed: int) -> Path:
    experiment_name = config.get("experiment", {}).get("name") or config["model"]["name"]
    run_dir = output_root / experiment_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_name = config["model"]["name"]
    model_cls = MODEL_REGISTRY.get(model_name)
    return model_cls(config)


def enrich_loss_config_with_dataset_statistics(config: dict[str, Any], train_loader) -> dict[str, Any]:
    cfg = dict(config)
    loss_cfg = dict(cfg.get("loss", {}))
    dataset = train_loader.dataset

    class_counts = getattr(dataset, "class_counts", None)
    class_weights = getattr(dataset, "class_weights", None)
    loss_name = loss_cfg.get("name")

    if loss_name in {"weighted_cross_entropy", "focal", "ldam"} and class_counts is not None:
        loss_cfg.setdefault("class_counts", class_counts)

    if loss_name in {"weighted_cross_entropy", "focal"}:
        if "class_weights" not in loss_cfg:
            loss_cfg["class_weights"] = class_weights
        elif loss_cfg["class_weights"] == "balanced":
            loss_cfg["class_weights"] = class_weights

    cfg["loss"] = loss_cfg
    cfg["train_dataset_stats"] = {
        "class_counts": class_counts,
        "class_weights": class_weights,
    }
    return cfg


def save_metrics_json(
    output_path: Path,
    fit_result: dict[str, Any],
    valid_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> None:
    payload = {
        "fit": fit_result,
        "valid": valid_metrics,
        "test": test_metrics,
    }
    write_json(payload, output_path)


def main() -> int:
    args = build_parser().parse_args()

    config = resolve_experiment_config(args.config)
    config = apply_cli_overrides(config, args)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    run_dir = build_run_dir(config, args.output_root, seed)

    processor = build_speech_processor(config["model"])
    train_loader, valid_loader, test_loader = build_speech_dataloaders(
        config=config,
        processor=processor,
    )

    config = enrich_loss_config_with_dataset_statistics(config, train_loader)
    write_yaml(config, run_dir / "resolved_config.yaml")
    write_json(config["train_dataset_stats"], run_dir / "train_dataset_stats.json")

    model = build_model(config)
    if args.resume_from is not None:
        ckpt = load_checkpoint(args.resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

    label_names = config["label_space"]["labels"]

    trainer = SpeechTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        class_names=label_names,
        run_dir=run_dir,
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

    test_pred_df = epoch_result_to_dataframe(test_result, label_names=label_names)
    save_basic_error_analysis(
        pred_df=test_pred_df,
        output_dir=run_dir / "error_analysis",
        split_name="test",
        max_rows=int(config.get("output", {}).get("max_error_rows", 200)),
    )

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
    )

    print(f"[OK] Finished speech run: {run_dir}")
    print(json.dumps(test_result.metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())