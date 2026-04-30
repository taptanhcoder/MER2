from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "sample_id",
    "label_id",
    "pred_id",
    "confidence",
    "entropy",
    "top2_margin",
    "probs",
    "logits",
    "split",
]


def _parse_json_vector(value: Any) -> list[float]:
    if isinstance(value, list):
        return [float(x) for x in value]

    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected JSON list, got: {type(parsed)}")
        return [float(x) for x in parsed]

    raise TypeError(f"Unsupported vector cell type: {type(value)}")


def stack_vector_column(series: pd.Series, column_name: str) -> np.ndarray:
    vectors = [_parse_json_vector(value) for value in series.tolist()]
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)

    vector_lengths = {len(vec) for vec in vectors}
    if len(vector_lengths) != 1:
        raise ValueError(
            f"Column '{column_name}' contains inconsistent vector lengths: {sorted(vector_lengths)}"
        )

    return np.asarray(vectors, dtype=np.float32)


def load_prediction_frame(csv_path: str | Path) -> pd.DataFrame:
    file_path = Path(csv_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {file_path}")

    frame = pd.read_csv(file_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required prediction columns: {missing}")
    return frame


def extract_logits_and_labels(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    logits = stack_vector_column(frame["logits"], "logits")
    labels = frame["label_id"].to_numpy(dtype=np.int64)
    return logits, labels


def top2_margin_from_probs(probs: np.ndarray) -> np.ndarray:
    if probs.size == 0:
        return np.zeros((0,), dtype=np.float32)

    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    if sorted_probs.shape[1] == 1:
        return sorted_probs[:, 0].astype(np.float32)
    return (sorted_probs[:, 0] - sorted_probs[:, 1]).astype(np.float32)


def _infer_label_id_to_name(frame: pd.DataFrame) -> dict[int, str]:
    if "label_id" not in frame.columns or "label" not in frame.columns:
        return {}

    pairs = (
        frame[["label_id", "label"]]
        .dropna()
        .drop_duplicates()
        .sort_values("label_id")
    )

    mapping: dict[int, str] = {}
    for _, row in pairs.iterrows():
        label_id = int(row["label_id"])
        label_name = str(row["label"])
        if label_id in mapping and mapping[label_id] != label_name:
            return {}
        mapping[label_id] = label_name
    return mapping


def build_calibrated_prediction_frame(
    original_frame: pd.DataFrame,
    calibrated_logits: np.ndarray,
    method_name: str,
    temperature: float,
) -> pd.DataFrame:
    calibrated_probs = np.asarray(calibrated_logits, dtype=np.float64)
    calibrated_probs = np.exp(
        calibrated_probs - calibrated_probs.max(axis=1, keepdims=True)
    )
    calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)

    calibrated_pred_id = calibrated_probs.argmax(axis=1).astype(np.int64)
    calibrated_confidence = calibrated_probs.max(axis=1).astype(np.float64)
    eps = 1e-12
    calibrated_entropy = (
        -(calibrated_probs * np.log(np.clip(calibrated_probs, eps, 1.0))).sum(axis=1)
    ).astype(np.float64)
    calibrated_margin = top2_margin_from_probs(calibrated_probs).astype(np.float64)

    frame = original_frame.copy()

    # Preserve original uncalibrated outputs for audit/debug
    frame["pred_id_uncalibrated"] = frame["pred_id"]
    frame["confidence_uncalibrated"] = frame["confidence"]
    frame["entropy_uncalibrated"] = frame["entropy"]
    frame["top2_margin_uncalibrated"] = frame["top2_margin"]
    frame["probs_uncalibrated"] = frame["probs"]
    frame["logits_uncalibrated"] = frame["logits"]

    if "pred" in frame.columns:
        frame["pred_uncalibrated"] = frame["pred"]

    frame["pred_id"] = calibrated_pred_id
    frame["confidence"] = calibrated_confidence
    frame["entropy"] = calibrated_entropy
    frame["top2_margin"] = calibrated_margin
    frame["probs"] = [
        json.dumps(row.tolist(), ensure_ascii=False) for row in calibrated_probs
    ]
    frame["logits"] = [
        json.dumps(row.tolist(), ensure_ascii=False) for row in calibrated_logits
    ]
    frame["calibration_method"] = method_name
    frame["temperature"] = float(temperature)

    label_map = _infer_label_id_to_name(frame)
    if "pred" in frame.columns and label_map:
        frame["pred"] = frame["pred_id"].map(label_map).fillna(frame["pred"])

    return frame