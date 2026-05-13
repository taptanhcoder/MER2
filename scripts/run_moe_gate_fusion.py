from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.extensions.moe_gate import (
    MoEGateConfig,
    alpha_summary,
    build_moe_features,
    classification_metrics,
    fit_standardizer,
    get_expert_logits,
    get_labels,
    get_sample_ids,
    load_torch_artifact,
    predict_moe_gate,
    save_json,
    save_predictions_csv,
    train_moe_gate,
)
from src.utils.config import load_yaml
from src.utils.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MoE Gating extension experiments on frozen fusion artifacts."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="MoE Gating extension config.",
    )
    return parser


def _resolve(path: str | Path) -> Path:
    return resolve_project_path(path, start=PROJECT_ROOT)


def _load_artifacts(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    artifacts_cfg = config["dataset"]["fusion_artifacts"]
    artifacts = {}

    for split in ["train", "valid", "test"]:
        if split not in artifacts_cfg:
            raise ValueError(f"Missing dataset.fusion_artifacts.{split}")
        artifacts[split] = load_torch_artifact(_resolve(artifacts_cfg[split]))

    return artifacts


def _feature_kwargs(method_cfg: dict[str, Any]) -> dict[str, Any]:
    features_cfg = dict(method_cfg.get("features", {}))

    return {
        "use_logits": bool(features_cfg.get("use_logits", True)),
        "use_probs": bool(features_cfg.get("use_probs", True)),
        "use_reliability": bool(features_cfg.get("use_reliability", True)),
        "use_derived": bool(features_cfg.get("use_derived", True)),
        "use_agreement": bool(features_cfg.get("use_agreement", True)),
        "use_pred_onehot": bool(features_cfg.get("use_pred_onehot", False)),
        "prefer_calibrated": bool(features_cfg.get("prefer_calibrated", True)),
    }


def _build_split_payloads(
    artifacts: dict[str, dict[str, Any]],
    method_cfg: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    kwargs = _feature_kwargs(method_cfg)
    payloads: dict[str, dict[str, Any]] = {}

    for split, artifact in artifacts.items():
        features, labels, feature_names = build_moe_features(
            artifact=artifact,
            **kwargs,
        )

        text_logits = get_expert_logits(
            artifact,
            "text",
            prefer_calibrated=kwargs["prefer_calibrated"],
        )
        speech_logits = get_expert_logits(
            artifact,
            "speech",
            prefer_calibrated=kwargs["prefer_calibrated"],
        )

        payloads[split] = {
            "features": features,
            "labels": labels,
            "feature_names": feature_names,
            "text_logits": text_logits,
            "speech_logits": speech_logits,
            "sample_ids": get_sample_ids(artifact),
        }

    return payloads


def _method_config_from_yaml(
    method_cfg: dict[str, Any],
    input_dim: int,
    seed: int,
) -> MoEGateConfig:
    model_cfg = dict(method_cfg.get("model", {}))

    return MoEGateConfig(
        input_dim=int(input_dim),
        num_classes=int(method_cfg.get("num_classes", 5)),
        gate_type=str(method_cfg.get("gate_type", "classwise")),
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        dropout=float(model_cfg.get("dropout", 0.10)),
        lr=float(model_cfg.get("lr", 1.0e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 1.0e-3)),
        epochs=int(model_cfg.get("epochs", 250)),
        patience=int(model_cfg.get("patience", 30)),
        batch_size=int(model_cfg.get("batch_size", 16)),
        seed=int(seed),
        gate_entropy_lambda=float(model_cfg.get("gate_entropy_lambda", 0.0)),
        gate_balance_lambda=float(model_cfg.get("gate_balance_lambda", 0.0)),
    )


def _run_method_seed(
    method_cfg: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    output_dir: Path,
    device: str,
    seed: int,
) -> dict[str, Any]:
    method_name = str(method_cfg["name"])
    method_seed_dir = output_dir / method_name / f"seed_{int(seed)}"
    method_seed_dir.mkdir(parents=True, exist_ok=True)

    scaler = fit_standardizer(payloads["train"]["features"])

    x_train = scaler.transform(payloads["train"]["features"])
    x_valid = scaler.transform(payloads["valid"]["features"])
    x_test = scaler.transform(payloads["test"]["features"])

    config = _method_config_from_yaml(
        method_cfg=method_cfg,
        input_dim=x_train.shape[1],
        seed=seed,
    )

    model, train_report = train_moe_gate(
        x_train=x_train,
        y_train=payloads["train"]["labels"],
        text_logits_train=payloads["train"]["text_logits"],
        speech_logits_train=payloads["train"]["speech_logits"],
        x_valid=x_valid,
        y_valid=payloads["valid"]["labels"],
        text_logits_valid=payloads["valid"]["text_logits"],
        speech_logits_valid=payloads["valid"]["speech_logits"],
        config=config,
        device=device,
    )

    valid_probs, valid_preds, valid_alpha = predict_moe_gate(
        model=model,
        x=x_valid,
        text_logits=payloads["valid"]["text_logits"],
        speech_logits=payloads["valid"]["speech_logits"],
        device=device,
    )
    test_probs, test_preds, test_alpha = predict_moe_gate(
        model=model,
        x=x_test,
        text_logits=payloads["test"]["text_logits"],
        speech_logits=payloads["test"]["speech_logits"],
        device=device,
    )

    valid_metrics = classification_metrics(
        payloads["valid"]["labels"],
        valid_preds,
    )
    test_metrics = classification_metrics(
        payloads["test"]["labels"],
        test_preds,
    )

    valid_alpha_summary = alpha_summary(valid_alpha)
    test_alpha_summary = alpha_summary(test_alpha)

    save_predictions_csv(
        output_path=method_seed_dir / "valid_predictions.csv",
        sample_ids=payloads["valid"]["sample_ids"],
        labels=payloads["valid"]["labels"],
        probs=valid_probs,
        alpha=valid_alpha,
        method_name=method_name,
        seed=seed,
    )
    save_predictions_csv(
        output_path=method_seed_dir / "test_predictions.csv",
        sample_ids=payloads["test"]["sample_ids"],
        labels=payloads["test"]["labels"],
        probs=test_probs,
        alpha=test_alpha,
        method_name=method_name,
        seed=seed,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "method": method_name,
            "seed": int(seed),
            "config": config.__dict__,
            "feature_names": payloads["train"]["feature_names"],
            "scaler": scaler.to_dict(),
            "train_report": train_report,
        },
        method_seed_dir / "model.pt",
    )

    save_json(
        {
            "method": method_name,
            "seed": int(seed),
            "feature_names": payloads["train"]["feature_names"],
            "scaler": scaler.to_dict(),
            "train_report": train_report,
            "valid": valid_metrics,
            "test": test_metrics,
            "valid_alpha": valid_alpha_summary,
            "test_alpha": test_alpha_summary,
        },
        method_seed_dir / "summary.json",
    )

    row: dict[str, Any] = {
        "method": method_name,
        "seed": int(seed),
        "gate_type": str(method_cfg.get("gate_type", "classwise")),
        "num_features": int(x_train.shape[1]),
        "best_valid_macro_f1": float(train_report["best_valid_macro_f1"]),
        "best_epoch": int(train_report["best_epoch"]),
    }

    for split_name, metrics in [("valid", valid_metrics), ("test", test_metrics)]:
        for metric_name, value in metrics.items():
            row[f"{split_name}_{metric_name}"] = float(value)

    for key, value in valid_alpha_summary.items():
        row[f"valid_{key}"] = value
    for key, value in test_alpha_summary.items():
        row[f"test_{key}"] = value

    return row


def _aggregate_rows(per_run: pd.DataFrame) -> pd.DataFrame:
    records = []

    for method_name, group in per_run.groupby("method", dropna=False):
        test_values = group["test_macro_f1"].astype(float)
        valid_values = group["valid_macro_f1"].astype(float)

        record = {
            "method": method_name,
            "num_runs": int(len(group)),
            "seeds": ",".join(str(int(seed)) for seed in group["seed"].tolist()),
            "test_macro_f1_mean": float(test_values.mean()),
            "test_macro_f1_std": float(test_values.std(ddof=0)),
            "test_macro_f1_min": float(test_values.min()),
            "test_macro_f1_max": float(test_values.max()),
            "test_macro_f1_range": float(test_values.max() - test_values.min()),
            "valid_macro_f1_mean": float(valid_values.mean()),
            "valid_macro_f1_std": float(valid_values.std(ddof=0)),
            "test_accuracy_mean": float(group["test_accuracy"].astype(float).mean()),
            "test_alpha_text_mean": float(group["test_alpha_text_mean"].astype(float).mean()),
            "test_alpha_speech_mean": float(group["test_alpha_speech_mean"].astype(float).mean()),
        }
        records.append(record)

    return pd.DataFrame(records).sort_values(
        ["test_macro_f1_mean", "test_macro_f1_std"],
        ascending=[False, True],
    )


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(_resolve(args.config))

    output_dir = _resolve(
        config.get("output_dir", "outputs/reports/extensions/moe_gate")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = str(config.get("device", "cuda"))
    artifacts = _load_artifacts(config)
    methods = list(config.get("methods", []))

    if not methods:
        raise ValueError("MoE Gate config must define at least one method.")

    per_run_rows = []

    for method_cfg in methods:
        if "name" not in method_cfg:
            raise ValueError("Every MoE method must have a `name`.")

        seeds = method_cfg.get("seeds", config.get("seeds", [42]))
        payloads = _build_split_payloads(
            artifacts=artifacts,
            method_cfg=method_cfg,
        )

        for seed in seeds:
            print(f"[RUN] method={method_cfg['name']} seed={seed}")
            row = _run_method_seed(
                method_cfg=method_cfg,
                payloads=payloads,
                output_dir=output_dir,
                device=device,
                seed=int(seed),
            )
            per_run_rows.append(row)

    per_run = pd.DataFrame(per_run_rows)
    aggregate = _aggregate_rows(per_run)

    per_run.to_csv(output_dir / "moe_gate_per_run_summary.csv", index=False)
    aggregate.to_csv(output_dir / "moe_gate_summary.csv", index=False)

    save_json(
        per_run.to_dict(orient="records"),
        output_dir / "moe_gate_per_run_summary.json",
    )
    save_json(
        aggregate.to_dict(orient="records"),
        output_dir / "moe_gate_summary.json",
    )

    print(f"[OK] Wrote MoE Gate extension results to: {output_dir}")
    print(json.dumps(aggregate.to_dict(orient="records"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())