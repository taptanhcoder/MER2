from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


@dataclass
class DiscreteNaiveBayesConfig:
    num_classes: int = 5
    alpha: float = 1.0


class DiscreteNaiveBayesClassifier:
    """
    Discrete Naive Bayes classifier for probabilistic decision-level fusion.

    This model estimates:

        P(Y | X_1, ..., X_d) ∝ P(Y) Π_i P(X_i | Y)

    All features must be integer-coded categorical values.
    Continuous reliability features should be discretized before fitting.
    """

    def __init__(self, config: DiscreteNaiveBayesConfig) -> None:
        self.config = config
        self.class_log_prior_: np.ndarray | None = None
        self.feature_log_probs_: list[np.ndarray] = []
        self.feature_cardinalities_: list[int] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DiscreteNaiveBayesClassifier":
        if x.ndim != 2:
            raise ValueError(f"Expected x shape [N, D], got {x.shape}")
        if y.ndim != 1:
            raise ValueError(f"Expected y shape [N], got {y.shape}")
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Mismatched number of rows: x={x.shape[0]}, y={y.shape[0]}"
            )

        num_classes = int(self.config.num_classes)
        alpha = float(self.config.alpha)

        y = y.astype("int64")
        class_counts = np.bincount(y, minlength=num_classes).astype("float64")

        class_prior = (class_counts + alpha) / (
            class_counts.sum() + alpha * num_classes
        )
        self.class_log_prior_ = np.log(class_prior)

        self.feature_log_probs_ = []
        self.feature_cardinalities_ = []

        for feature_idx in range(x.shape[1]):
            values = x[:, feature_idx].astype("int64")
            cardinality = int(values.max()) + 1
            cardinality = max(cardinality, 2)

            self.feature_cardinalities_.append(cardinality)

            counts = np.zeros((num_classes, cardinality), dtype="float64")
            for class_idx in range(num_classes):
                class_values = values[y == class_idx]
                if class_values.size > 0:
                    counts[class_idx] = np.bincount(
                        class_values,
                        minlength=cardinality,
                    )[:cardinality]

            probs = (counts + alpha) / (
                counts.sum(axis=1, keepdims=True) + alpha * cardinality
            )
            self.feature_log_probs_.append(np.log(probs))

        return self

    def predict_log_proba(self, x: np.ndarray) -> np.ndarray:
        if self.class_log_prior_ is None:
            raise RuntimeError("DiscreteNaiveBayesClassifier is not fitted.")

        log_probs = np.tile(self.class_log_prior_[None, :], (x.shape[0], 1))

        for feature_idx, table in enumerate(self.feature_log_probs_):
            values = x[:, feature_idx].astype("int64")
            values = np.clip(values, 0, table.shape[1] - 1)

            for class_idx in range(table.shape[0]):
                log_probs[:, class_idx] += table[class_idx, values]

        log_norm = np.logaddexp.reduce(log_probs, axis=1, keepdims=True)
        return log_probs - log_norm

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_proba(x).argmax(axis=1)

    def to_dict(self) -> dict[str, Any]:
        if self.class_log_prior_ is None:
            raise RuntimeError("Cannot serialize an unfitted model.")

        return {
            "config": {
                "num_classes": int(self.config.num_classes),
                "alpha": float(self.config.alpha),
            },
            "class_log_prior": self.class_log_prior_.tolist(),
            "feature_cardinalities": [int(x) for x in self.feature_cardinalities_],
            "feature_log_probs": [table.tolist() for table in self.feature_log_probs_],
        }


def load_torch_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Fusion artifact not found: {artifact_path}")
    return torch.load(artifact_path, map_location="cpu")


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -(probs * np.log(np.clip(probs, eps, 1.0))).sum(axis=1)


def _confidence(probs: np.ndarray) -> np.ndarray:
    return probs.max(axis=1)


def _margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    if sorted_probs.shape[1] == 1:
        return sorted_probs[:, -1]
    return sorted_probs[:, -1] - sorted_probs[:, -2]


def _pred_id(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(axis=1).astype("int64")


def get_expert_logits(artifact: dict[str, Any], prefix: str) -> np.ndarray:
    calibrated_key = f"{prefix}_logits_cal"
    raw_key = f"{prefix}_logits_raw"

    if calibrated_key in artifact:
        return _to_numpy(artifact[calibrated_key]).astype("float32")
    if raw_key in artifact:
        return _to_numpy(artifact[raw_key]).astype("float32")

    raise KeyError(f"Missing `{calibrated_key}` or `{raw_key}` in artifact.")


def get_expert_probs(artifact: dict[str, Any], prefix: str) -> np.ndarray:
    calibrated_key = f"{prefix}_probs_cal"
    raw_key = f"{prefix}_probs"

    if calibrated_key in artifact:
        return _to_numpy(artifact[calibrated_key]).astype("float32")
    if raw_key in artifact:
        return _to_numpy(artifact[raw_key]).astype("float32")

    logits = get_expert_logits(artifact, prefix)
    return _softmax(logits).astype("float32")


def _fit_quantile_edges(values: np.ndarray, num_bins: int = 3) -> np.ndarray:
    values = values.reshape(-1).astype("float64")
    if num_bins <= 1:
        return np.array([], dtype="float32")

    quantiles = [i / num_bins for i in range(1, num_bins)]
    edges = np.quantile(values, quantiles)

    # Degenerate case: all values are almost identical.
    # Use min/max fallback to avoid producing invalid bins.
    if np.allclose(edges, edges[0]):
        min_value = float(values.min())
        max_value = float(values.max())
        if math.isclose(min_value, max_value):
            edges = np.array([min_value], dtype="float32")
        else:
            edges = np.linspace(min_value, max_value, num_bins + 1)[1:-1]

    return edges.astype("float32")


def _digitize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = values.reshape(-1)
    return np.digitize(values, edges, right=False).astype("int64")


def build_bayes_features_from_artifact(
    artifact: dict[str, Any],
    variant: str,
    bin_edges: dict[str, np.ndarray] | None = None,
    num_bins: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str]]:
    """
    Build categorical features from frozen text/speech fusion artifacts.

    Supported variants:
    - pred_only:
        text_pred, speech_pred, text_speech_agreement

    - reliability:
        pred_only + confidence/entropy/margin bins for text/speech

    This function intentionally avoids duration/token length features by default
    to reduce shortcut risk.
    """
    variant = str(variant).strip().lower()
    if variant not in {"pred_only", "reliability"}:
        raise ValueError(
            f"Unsupported Bayes feature variant={variant!r}. "
            "Use `pred_only` or `reliability`."
        )

    labels = _to_numpy(artifact["label_id"]).astype("int64")

    text_probs = get_expert_probs(artifact, "text")
    speech_probs = get_expert_probs(artifact, "speech")

    text_pred = _pred_id(text_probs)
    speech_pred = _pred_id(speech_probs)
    agreement = (text_pred == speech_pred).astype("int64")

    columns = [
        text_pred.reshape(-1, 1),
        speech_pred.reshape(-1, 1),
        agreement.reshape(-1, 1),
    ]
    names = [
        "text_pred_id",
        "speech_pred_id",
        "text_speech_agreement",
    ]

    edges: dict[str, np.ndarray] = {} if bin_edges is None else dict(bin_edges)

    if variant == "reliability":
        continuous_features = {
            "text_confidence": _confidence(text_probs),
            "speech_confidence": _confidence(speech_probs),
            "text_entropy": _entropy(text_probs),
            "speech_entropy": _entropy(speech_probs),
            "text_margin": _margin(text_probs),
            "speech_margin": _margin(speech_probs),
        }

        for name, values in continuous_features.items():
            if name not in edges:
                edges[name] = _fit_quantile_edges(values, num_bins=num_bins)

            binned = _digitize(values, edges[name])
            columns.append(binned.reshape(-1, 1))
            names.append(f"{name}_bin")

    x = np.concatenate(columns, axis=1).astype("int64")
    return x, labels, edges, names


def build_bayes_features_from_prediction_csv(
    csv_path: str | Path,
    variant: str,
    bin_edges: dict[str, np.ndarray] | None = None,
    num_bins: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str], list[str]]:
    """
    Build categorical features from fusion prediction CSVs that contain branch outputs.

    This supports interaction-aware Bayes Net variants if the CSV has:
    - text_pred_id
    - speech_pred_id
    - interaction_pred_id
    - text_confidence
    - speech_confidence
    - interaction_confidence

    Supported variants:
    - pred_only
    - reliability
    - interaction_reliability
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")

    frame = pd.read_csv(path)
    variant = str(variant).strip().lower()

    required = ["sample_id", "label_id", "text_pred_id", "speech_pred_id"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required columns for Bayes Net: {missing}")

    if variant not in {"pred_only", "reliability", "interaction_reliability"}:
        raise ValueError(
            f"Unsupported CSV Bayes feature variant={variant!r}. "
            "Use pred_only, reliability, or interaction_reliability."
        )

    labels = frame["label_id"].astype("int64").to_numpy()
    sample_ids = frame["sample_id"].astype(str).tolist()

    text_pred = frame["text_pred_id"].astype("int64").to_numpy()
    speech_pred = frame["speech_pred_id"].astype("int64").to_numpy()
    text_speech_agreement = (text_pred == speech_pred).astype("int64")

    columns = [
        text_pred.reshape(-1, 1),
        speech_pred.reshape(-1, 1),
        text_speech_agreement.reshape(-1, 1),
    ]
    names = [
        "text_pred_id",
        "speech_pred_id",
        "text_speech_agreement",
    ]

    edges: dict[str, np.ndarray] = {} if bin_edges is None else dict(bin_edges)

    continuous_names = [
        "text_confidence",
        "speech_confidence",
    ]

    if variant in {"reliability", "interaction_reliability"}:
        for col in continuous_names:
            if col not in frame.columns:
                raise ValueError(f"{path} missing reliability column `{col}`.")

            values = frame[col].astype("float64").to_numpy()
            if col not in edges:
                edges[col] = _fit_quantile_edges(values, num_bins=num_bins)

            columns.append(_digitize(values, edges[col]).reshape(-1, 1))
            names.append(f"{col}_bin")

    if variant == "interaction_reliability":
        interaction_required = ["interaction_pred_id", "interaction_confidence"]
        missing_interaction = [
            col for col in interaction_required if col not in frame.columns
        ]
        if missing_interaction:
            raise ValueError(
                f"{path} missing interaction columns: {missing_interaction}"
            )

        interaction_pred = frame["interaction_pred_id"].astype("int64").to_numpy()
        text_interaction_agreement = (text_pred == interaction_pred).astype("int64")
        speech_interaction_agreement = (speech_pred == interaction_pred).astype("int64")

        columns.extend(
            [
                interaction_pred.reshape(-1, 1),
                text_interaction_agreement.reshape(-1, 1),
                speech_interaction_agreement.reshape(-1, 1),
            ]
        )
        names.extend(
            [
                "interaction_pred_id",
                "text_interaction_agreement",
                "speech_interaction_agreement",
            ]
        )

        values = frame["interaction_confidence"].astype("float64").to_numpy()
        if "interaction_confidence" not in edges:
            edges["interaction_confidence"] = _fit_quantile_edges(
                values,
                num_bins=num_bins,
            )
        columns.append(
            _digitize(values, edges["interaction_confidence"]).reshape(-1, 1)
        )
        names.append("interaction_confidence_bin")

    x = np.concatenate(columns, axis=1).astype("int64")
    return x, labels, edges, names, sample_ids


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


def save_predictions_csv(
    output_path: str | Path,
    sample_ids: list[str],
    labels: np.ndarray,
    probs: np.ndarray,
    method_name: str,
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
            }
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_feature_table(
    output_path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> None:
    frame = pd.DataFrame(x, columns=feature_names)
    frame.insert(0, "label_id", y.astype("int64"))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)