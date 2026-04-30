from __future__ import annotations

import json

import pandas as pd

from src.text.projection import (
    compute_projected_mer5_metrics,
    project_prediction_dataframe_to_mer5,
    project_prob_vector_to_mer5,
)


def test_project_prob_vector_to_mer5_drop_policy() -> None:
    native_labels = [
        "anger",
        "fear",
        "enjoyment",
        "sadness",
        "disgust",
        "surprise",
        "other",
    ]
    probs = [0.1, 0.2, 0.3, 0.1, 0.2, 0.0, 0.1]

    projected, unsupported_mass = project_prob_vector_to_mer5(
        prob_vector=probs,
        native_label_names=native_labels,
        unsupported_policy="drop",
    )

    assert len(projected) == 5
    assert abs(sum(projected) - 1.0) < 1e-6
    assert abs(unsupported_mass - 0.2) < 1e-6


def test_project_prediction_dataframe_to_mer5() -> None:
    native_labels = [
        "anger",
        "fear",
        "enjoyment",
        "sadness",
        "disgust",
        "surprise",
        "other",
    ]
    df = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "label_id": 2,
                "pred_id": 2,
                "label": "enjoyment",
                "pred": "enjoyment",
                "probs": json.dumps([0.1, 0.1, 0.5, 0.1, 0.0, 0.0, 0.2]),
            },
            {
                "sample_id": "b",
                "label_id": 4,
                "pred_id": 4,
                "label": "disgust",
                "pred": "disgust",
                "probs": json.dumps([0.1, 0.1, 0.1, 0.1, 0.5, 0.0, 0.1]),
            },
        ]
    )

    projected_df = project_prediction_dataframe_to_mer5(
        pred_df=df,
        native_label_names=native_labels,
        unsupported_policy="drop",
    )

    assert "fusion_probs" in projected_df.columns
    assert "fusion_pred" in projected_df.columns
    assert "unsupported_prob_mass" in projected_df.columns
    assert projected_df.iloc[0]["fusion_label"] == "happiness"
    assert projected_df.iloc[1]["fusion_label"] is None or pd.isna(projected_df.iloc[1]["fusion_label"])


def test_compute_projected_mer5_metrics() -> None:
    native_labels = [
        "anger",
        "fear",
        "enjoyment",
        "sadness",
        "disgust",
        "surprise",
        "other",
    ]
    df = pd.DataFrame(
        [
            {
                "sample_id": "a",
                "label_id": 2,
                "pred_id": 2,
                "label": "enjoyment",
                "pred": "enjoyment",
                "probs": json.dumps([0.1, 0.1, 0.6, 0.1, 0.0, 0.0, 0.1]),
            },
            {
                "sample_id": "b",
                "label_id": 0,
                "pred_id": 0,
                "label": "anger",
                "pred": "anger",
                "probs": json.dumps([0.6, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1]),
            },
            {
                "sample_id": "c",
                "label_id": 4,
                "pred_id": 4,
                "label": "disgust",
                "pred": "disgust",
                "probs": json.dumps([0.1, 0.1, 0.1, 0.1, 0.5, 0.0, 0.1]),
            },
        ]
    )

    projected_df = project_prediction_dataframe_to_mer5(
        pred_df=df,
        native_label_names=native_labels,
        unsupported_policy="drop",
    )
    metrics = compute_projected_mer5_metrics(projected_df)

    assert metrics["num_rows"] == 3
    assert metrics["num_supported_rows"] == 2
    assert metrics["num_dropped_rows"] == 1
    assert "macro_f1" in metrics["metrics"]