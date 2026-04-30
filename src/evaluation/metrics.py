from __future__ import annotations

from typing import Any, Sequence

import numpy as np # type: ignore
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    labels = (
        np.unique(np.concatenate([y_true_arr, y_pred_arr]))
        if len(y_true_arr)
        else np.array([], dtype=int)
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=labels,
        zero_division=0,
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for idx, label in enumerate(labels.tolist()):
        class_key = class_names[label] if class_names and label < len(class_names) else str(label)
        per_class[class_key] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }

    macro_recall = float(np.mean(recall)) if len(recall) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)) if len(y_true_arr) else 0.0,
        "macro_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        )
        if len(y_true_arr)
        else 0.0,
        "weighted_f1": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        )
        if len(y_true_arr)
        else 0.0,
        "macro_precision": float(np.mean(precision)) if len(precision) else 0.0,
        "macro_recall": macro_recall,
        "uar": float(balanced_accuracy_score(y_true_arr, y_pred_arr))
        if len(y_true_arr)
        else 0.0,
        "per_class": per_class,
    }