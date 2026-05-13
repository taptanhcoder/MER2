from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.extensions.xgboost_fusion import (
    build_xgboost_classifier,
    build_xgboost_features,
    classification_metrics,
    fit_xgboost_classifier,
    load_torch_artifact,
    predict_proba_safe,
    save_feature_importance,
    save_json,
    save_predictions_csv,
)
from src.fusion.extensions.moe_gate import get_labels, get_sample_ids
from src.utils.config import load_yaml
from src.utils.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run XGBoost tabular meta-fusion extension experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="XGBoost fusion extension config.",
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
        features, labels, feature_names = build_xgboost_features(
            artifact=artifact,
            **kwargs,
        )

        payloads[split] = {
            "features": features,
            "labels": labels,
            "feature_names": feature_names,
            "sample_ids": get_sample_ids(artifact),
        }

    return payloads


def _run_method_seed(
    method_cfg: dict[str, Any],
    payloads: dict[str, dict[str, Any]],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    method_name = str(method_cfg["name"])
    num_classes = int(method_cfg.get("num_classes", 5))

    method_seed_dir = output_dir / method_name / f"seed_{int(seed)}"
    method_seed_dir.mkdir(parents=True, exist_ok=True)

    x_train = payloads["train"]["features"]
    y_train = payloads["train"]["labels"]

    x_valid = payloads["valid"]["features"]
    y_valid = payloads["valid"]["labels"]

    x_test = payloads["test"]["features"]
    y_test = payloads["test"]["labels"]

    model = build_xgboost_classifier(
        method_cfg=method_cfg,
        seed=int(seed),
    )
    model = fit_xgboost_classifier(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_valid=x_valid,
        y_valid=y_valid,
    )

    valid_probs = predict_proba_safe(
        model=model,
        x=x_valid,
        num_classes=num_classes,
    )
    test_probs = predict_proba_safe(
        model=model,
        x=x_test,
        num_classes=num_classes,
    )

    valid_preds = valid_probs.argmax(axis=1)
    test_preds = test_probs.argmax(axis=1)

    valid_metrics = classification_metrics(y_valid, valid_preds)
    test_metrics = classification_metrics(y_test, test_preds)

    save_predictions_csv(
        output_path=method_seed_dir / "valid_predictions.csv",
        sample_ids=payloads["valid"]["sample_ids"],
        labels=y_valid,
        probs=valid_probs,
        method_name=method_name,
        seed=int(seed),
    )
    save_predictions_csv(
        output_path=method_seed_dir / "test_predictions.csv",
        sample_ids=payloads["test"]["sample_ids"],
        labels=y_test,
        probs=test_probs,
        method_name=method_name,
        seed=int(seed),
    )

    save_feature_importance(
        output_path=method_seed_dir / "feature_importance.csv",
        model=model,
        feature_names=payloads["train"]["feature_names"],
    )

    model_path = method_seed_dir / "model.json"
    try:
        model.save_model(str(model_path))
    except Exception:
        model_path = None

    summary = {
        "method": method_name,
        "seed": int(seed),
        "num_features": int(x_train.shape[1]),
        "feature_names": payloads["train"]["feature_names"],
        "model_path": str(model_path) if model_path is not None else None,
        "valid": valid_metrics,
        "test": test_metrics,
    }
    save_json(summary, method_seed_dir / "summary.json")

    row: dict[str, Any] = {
        "method": method_name,
        "seed": int(seed),
        "num_features": int(x_train.shape[1]),
    }

    for split_name, metrics in [("valid", valid_metrics), ("test", test_metrics)]:
        for metric_name, value in metrics.items():
            row[f"{split_name}_{metric_name}"] = float(value)

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
            "valid_test_gap_mean": float(
                valid_values.mean() - test_values.mean()
            ),
            "test_accuracy_mean": float(group["test_accuracy"].astype(float).mean()),
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
        config.get("output_dir", "outputs/reports/extensions/xgboost")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = _load_artifacts(config)
    methods = list(config.get("methods", []))

    if not methods:
        raise ValueError("XGBoost config must define at least one method.")

    per_run_rows = []

    for method_cfg in methods:
        if "name" not in method_cfg:
            raise ValueError("Every XGBoost method must have a `name`.")

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
                seed=int(seed),
            )
            per_run_rows.append(row)

    per_run = pd.DataFrame(per_run_rows)
    aggregate = _aggregate_rows(per_run)

    per_run.to_csv(output_dir / "xgboost_per_run_summary.csv", index=False)
    aggregate.to_csv(output_dir / "xgboost_summary.csv", index=False)

    save_json(
        per_run.to_dict(orient="records"),
        output_dir / "xgboost_per_run_summary.json",
    )
    save_json(
        aggregate.to_dict(orient="records"),
        output_dir / "xgboost_summary.json",
    )

    print(f"[OK] Wrote XGBoost extension results to: {output_dir}")
    print(json.dumps(aggregate.to_dict(orient="records"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())