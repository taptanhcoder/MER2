from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.calibration.io import (
    build_calibrated_prediction_frame,
    extract_logits_and_labels,
    load_prediction_frame,
)
from src.calibration.temperature_scaling import TemperatureScaler


def test_calibration_io_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "valid_predictions.csv"
    df = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "label_id": 0,
                "label": "anger",
                "pred_id": 0,
                "pred": "anger",
                "confidence": 0.90,
                "entropy": 0.20,
                "top2_margin": 0.70,
                "probs": "[0.9, 0.1]",
                "logits": "[2.2, 0.1]",
                "split": "valid",
            },
            {
                "sample_id": "b",
                "label_id": 1,
                "label": "fear",
                "pred_id": 0,
                "pred": "anger",
                "confidence": 0.55,
                "entropy": 0.69,
                "top2_margin": 0.10,
                "probs": "[0.55, 0.45]",
                "logits": "[0.7, 0.5]",
                "split": "valid",
            },
        ]
    )
    df.to_csv(csv_path, index=False)

    frame = load_prediction_frame(csv_path)
    logits, labels = extract_logits_and_labels(frame)

    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    calibrated_logits = scaler.transform_logits(logits)

    calibrated_frame = build_calibrated_prediction_frame(
        original_frame=frame,
        calibrated_logits=calibrated_logits,
        method_name="temperature_scaling",
        temperature=scaler.temperature,
    )

    assert "calibration_method" in calibrated_frame.columns
    assert "temperature" in calibrated_frame.columns
    assert "pred_id_uncalibrated" in calibrated_frame.columns
    assert "confidence_uncalibrated" in calibrated_frame.columns
    assert "entropy_uncalibrated" in calibrated_frame.columns
    assert "top2_margin_uncalibrated" in calibrated_frame.columns
    assert "probs_uncalibrated" in calibrated_frame.columns
    assert "logits_uncalibrated" in calibrated_frame.columns
    assert "pred_uncalibrated" in calibrated_frame.columns
    assert calibrated_frame["split"].tolist() == ["valid", "valid"]
    assert calibrated_frame["probs"].notna().all()
    assert calibrated_frame["logits"].notna().all()