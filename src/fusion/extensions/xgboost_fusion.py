from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.fusion.extensions.moe_gate import (
    build_moe_features,
    get_labels,
    get_sample_ids,
    load_torch_artifact,
)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def build_xgboost_features(
    artifact: dict[str, Any],
    use_logits: bool = True,
    use_probs: bool = True,
    use_reliability: bool = True,
    use_derived: bool = True,
    use_agreement: bool = True,
    use_pred_onehot: bool = False,
    prefer_calibrated: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build tabular meta-fusion features from frozen text/speech fusion artifacts.

    This intentionally reuses the same feature construction as MoE extension
    so Bayes Net / MoE / XGBoost can be compared on a controlled input source.
    """
    features, labels, feature_names = build_moe_features(
        artifact=artifact,
        use_logits=use_logits,
        use_probs=use_probs,
        use_reliability=use_reliability,
        use_derived=use_derived,
        use_agreement=use_agreement,
        use_pred_onehot=use_pred_onehot,
        prefer_calibrated=prefer_calibrated,
    )
    return features.astype("float32"), labels.astype("int64"), feature_names


def build_xgboost_classifier(
    method_cfg: dict[str, Any],
    seed: int,
):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "xgboost is not installed in the current environment. "
            "Install it first, for example: pip install xgboost"
        ) from exc

    model_cfg = dict(method_cfg.get("model", {}))

    return XGBClassifier(
        objective="multi:softprob",
        num_class=int(method_cfg.get("num_classes", 5)),
        eval_metric=str(model_cfg.get("eval_metric", "mlogloss")),
        n_estimators=int(model_cfg.get("n_estimators", 160)),
        max_depth=int(model_cfg.get("max_depth", 2)),
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        subsample=float(model_cfg.get("subsample", 0.80)),
        colsample_bytree=float(model_cfg.get("colsample_bytree", 0.80)),
        min_child_weight=float(model_cfg.get("min_child_weight", 2.0)),
        reg_lambda=float(model_cfg.get("reg_lambda", 2.0)),
        reg_alpha=float(model_cfg.get("reg_alpha", 0.0)),
        gamma=float(model_cfg.get("gamma", 0.0)),
        tree_method=str(model_cfg.get("tree_method", "hist")),
        random_state=int(seed),
        n_jobs=int(model_cfg.get("n_jobs", 4)),
        verbosity=int(model_cfg.get("verbosity", 0)),
    )


def fit_xgboost_classifier(
    model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray | None = None,
    y_valid: np.ndarray | None = None,
):
    """
    Fit with a conservative API path. XGBoost sklearn wrapper has changed
    early-stopping signatures across versions, so this function avoids
    version-fragile early stopping by default.
    """
    if x_valid is not None and y_valid is not None:
        try:
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                verbose=False,
            )
            return model
        except TypeError:
            pass

    model.fit(x_train, y_train)
    return model


def predict_proba_safe(model, x: np.ndarray, num_classes: int) -> np.ndarray:
    probs = model.predict_proba(x)

    if isinstance(probs, list):
        probs = np.stack(probs, axis=1)

    probs = np.asarray(probs, dtype="float32")

    if probs.ndim != 2:
        raise ValueError(f"Expected probability matrix [N, C], got shape={probs.shape}")

    if probs.shape[1] == num_classes:
        return probs

    # Defensive fallback in case an implementation returns fewer columns.
    fixed = np.zeros((probs.shape[0], num_classes), dtype="float32")
    fixed[:, : min(num_classes, probs.shape[1])] = probs[:, :num_classes]
    row_sum = fixed.sum(axis=1, keepdims=True)
    fixed = fixed / np.clip(row_sum, 1e-12, None)
    return fixed


def save_predictions_csv(
    output_path: str | Path,
    sample_ids: list[str],
    labels: np.ndarray,
    probs: np.ndarray,
    method_name: str,
    seed: int,
) -> None:
    preds = probs.argmax(axis=1)

    rows = []
    for idx, sample_id in enumerate(sample_ids):
        rows.append(
            {
                "sample_id": str(sample_id),
                "label_id": int(labels[idx]),
                "pred_id": int(preds[idx]),
                "confidence": float(probs[idx].max()),
                "probs": json.dumps(probs[idx].tolist(), ensure_ascii=False),
                "method": method_name,
                "seed": int(seed),
            }
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_feature_importance(
    output_path: str | Path,
    model,
    feature_names: list[str],
) -> None:
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return

    frame = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": np.asarray(importance, dtype="float32"),
        }
    ).sort_values("importance", ascending=False)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def save_json(data: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )