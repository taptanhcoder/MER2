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

from src.fusion.extensions.bayes_net import (
    DiscreteNaiveBayesClassifier,
    DiscreteNaiveBayesConfig,
    build_bayes_features_from_artifact,
    build_bayes_features_from_prediction_csv,
    classification_metrics,
    load_torch_artifact,
    save_feature_table,
    save_predictions_csv,
)
from src.utils.config import load_yaml
from src.utils.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Bayesian Network / Naive Bayes fusion extension experiments."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Bayes Net extension config.",
    )
    return parser


def _load_artifact_inputs(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config["dataset"]
    artifacts_cfg = dataset_cfg["fusion_artifacts"]

    return {
        split: load_torch_artifact(resolve_project_path(path, PROJECT_ROOT))
        for split, path in artifacts_cfg.items()
        if split in {"train", "valid", "test"}
    }


def _run_artifact_method(
    method_cfg: dict[str, Any],
    artifacts: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    method_name = str(method_cfg["name"])
    variant = str(method_cfg.get("variant", "pred_only"))
    alpha = float(method_cfg.get("alpha", 1.0))
    num_classes = int(method_cfg.get("num_classes", 5))
    num_bins = int(method_cfg.get("num_bins", 3))
    save_features = bool(method_cfg.get("save_features", True))

    x_train, y_train, bin_edges, feature_names = build_bayes_features_from_artifact(
        artifacts["train"],
        variant=variant,
        bin_edges=None,
        num_bins=num_bins,
    )
    x_valid, y_valid, _, _ = build_bayes_features_from_artifact(
        artifacts["valid"],
        variant=variant,
        bin_edges=bin_edges,
        num_bins=num_bins,
    )
    x_test, y_test, _, _ = build_bayes_features_from_artifact(
        artifacts["test"],
        variant=variant,
        bin_edges=bin_edges,
        num_bins=num_bins,
    )

    model = DiscreteNaiveBayesClassifier(
        DiscreteNaiveBayesConfig(
            num_classes=num_classes,
            alpha=alpha,
        )
    )
    model.fit(x_train, y_train)

    valid_probs = model.predict_proba(x_valid)
    test_probs = model.predict_proba(x_test)

    valid_preds = valid_probs.argmax(axis=1)
    test_preds = test_probs.argmax(axis=1)

    if save_features:
        save_feature_table(
            output_dir / f"{method_name}_train_features.csv",
            x_train,
            y_train,
            feature_names,
        )
        save_feature_table(
            output_dir / f"{method_name}_valid_features.csv",
            x_valid,
            y_valid,
            feature_names,
        )
        save_feature_table(
            output_dir / f"{method_name}_test_features.csv",
            x_test,
            y_test,
            feature_names,
        )

    save_predictions_csv(
        output_path=output_dir / f"{method_name}_valid_predictions.csv",
        sample_ids=list(artifacts["valid"]["sample_id"]),
        labels=y_valid,
        probs=valid_probs,
        method_name=method_name,
    )
    save_predictions_csv(
        output_path=output_dir / f"{method_name}_test_predictions.csv",
        sample_ids=list(artifacts["test"]["sample_id"]),
        labels=y_test,
        probs=test_probs,
        method_name=method_name,
    )

    model_payload = {
        "model": model.to_dict(),
        "feature_names": feature_names,
        "bin_edges": {key: value.tolist() for key, value in bin_edges.items()},
    }
    (output_dir / f"{method_name}_model.json").write_text(
        json.dumps(model_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "method": method_name,
        "source": "fusion_artifact",
        "variant": variant,
        "alpha": alpha,
        "num_bins": num_bins,
        "feature_names": feature_names,
        "valid": classification_metrics(y_valid, valid_preds),
        "test": classification_metrics(y_test, test_preds),
    }


def _run_prediction_csv_method(
    method_cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    method_name = str(method_cfg["name"])
    variant = str(method_cfg.get("variant", "interaction_reliability"))
    alpha = float(method_cfg.get("alpha", 1.0))
    num_classes = int(method_cfg.get("num_classes", 5))
    num_bins = int(method_cfg.get("num_bins", 3))
    save_features = bool(method_cfg.get("save_features", True))

    train_csv = resolve_project_path(method_cfg["train_csv"], PROJECT_ROOT)
    valid_csv = resolve_project_path(method_cfg["valid_csv"], PROJECT_ROOT)
    test_csv = resolve_project_path(method_cfg["test_csv"], PROJECT_ROOT)

    x_train, y_train, bin_edges, feature_names, train_ids = (
        build_bayes_features_from_prediction_csv(
            train_csv,
            variant=variant,
            bin_edges=None,
            num_bins=num_bins,
        )
    )
    x_valid, y_valid, _, _, valid_ids = build_bayes_features_from_prediction_csv(
        valid_csv,
        variant=variant,
        bin_edges=bin_edges,
        num_bins=num_bins,
    )
    x_test, y_test, _, _, test_ids = build_bayes_features_from_prediction_csv(
        test_csv,
        variant=variant,
        bin_edges=bin_edges,
        num_bins=num_bins,
    )

    model = DiscreteNaiveBayesClassifier(
        DiscreteNaiveBayesConfig(
            num_classes=num_classes,
            alpha=alpha,
        )
    )
    model.fit(x_train, y_train)

    valid_probs = model.predict_proba(x_valid)
    test_probs = model.predict_proba(x_test)

    valid_preds = valid_probs.argmax(axis=1)
    test_preds = test_probs.argmax(axis=1)

    if save_features:
        save_feature_table(
            output_dir / f"{method_name}_train_features.csv",
            x_train,
            y_train,
            feature_names,
        )
        save_feature_table(
            output_dir / f"{method_name}_valid_features.csv",
            x_valid,
            y_valid,
            feature_names,
        )
        save_feature_table(
            output_dir / f"{method_name}_test_features.csv",
            x_test,
            y_test,
            feature_names,
        )

    save_predictions_csv(
        output_path=output_dir / f"{method_name}_valid_predictions.csv",
        sample_ids=valid_ids,
        labels=y_valid,
        probs=valid_probs,
        method_name=method_name,
    )
    save_predictions_csv(
        output_path=output_dir / f"{method_name}_test_predictions.csv",
        sample_ids=test_ids,
        labels=y_test,
        probs=test_probs,
        method_name=method_name,
    )

    model_payload = {
        "model": model.to_dict(),
        "feature_names": feature_names,
        "bin_edges": {key: value.tolist() for key, value in bin_edges.items()},
        "train_csv": str(train_csv),
        "valid_csv": str(valid_csv),
        "test_csv": str(test_csv),
    }
    (output_dir / f"{method_name}_model.json").write_text(
        json.dumps(model_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "method": method_name,
        "source": "prediction_csv",
        "variant": variant,
        "alpha": alpha,
        "num_bins": num_bins,
        "feature_names": feature_names,
        "valid": classification_metrics(y_valid, valid_preds),
        "test": classification_metrics(y_test, test_preds),
    }


def main() -> int:
    args = build_parser().parse_args()
    config = load_yaml(resolve_project_path(args.config, PROJECT_ROOT))

    output_dir = resolve_project_path(
        config.get("output_dir", "outputs/reports/extensions/bayes_net"),
        PROJECT_ROOT,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = list(config.get("methods", []))
    if not methods:
        raise ValueError("Bayes Net config must contain at least one method.")

    artifacts = None
    if "dataset" in config and "fusion_artifacts" in config["dataset"]:
        artifacts = _load_artifact_inputs(config)

    summaries: list[dict[str, Any]] = []

    for method_cfg in methods:
        source = str(method_cfg.get("source", "fusion_artifact"))
        if source == "fusion_artifact":
            if artifacts is None:
                raise ValueError(
                    f"Method {method_cfg.get('name')} requires dataset.fusion_artifacts."
                )
            summary = _run_artifact_method(
                method_cfg=method_cfg,
                artifacts=artifacts,
                output_dir=output_dir,
            )
        elif source == "prediction_csv":
            summary = _run_prediction_csv_method(
                method_cfg=method_cfg,
                output_dir=output_dir,
            )
        else:
            raise ValueError(
                f"Unsupported Bayes Net method source={source!r}. "
                "Use `fusion_artifact` or `prediction_csv`."
            )

        summaries.append(summary)

    summary_path = output_dir / "bayes_net_summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    flat_rows = []
    for item in summaries:
        row = {
            "method": item["method"],
            "source": item["source"],
            "variant": item["variant"],
            "alpha": item["alpha"],
            "num_bins": item["num_bins"],
        }
        for split in ["valid", "test"]:
            for metric_name, metric_value in item[split].items():
                row[f"{split}_{metric_name}"] = metric_value
        flat_rows.append(row)

    pd.DataFrame(flat_rows).to_csv(output_dir / "bayes_net_summary.csv", index=False)

    print(f"[OK] Wrote Bayes Net extension results to: {output_dir}")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())