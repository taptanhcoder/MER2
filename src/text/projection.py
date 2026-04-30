from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.data.label_space import MER5_LABEL2ID, MER5_LABELS, UIT_VSMEC_NATIVE_TO_MER5
from src.evaluation.metrics import compute_classification_metrics


def _parse_probs(value: Any) -> list[float]:
    if isinstance(value, str):
        parsed = json.loads(value)
        return [float(x) for x in parsed]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    raise TypeError(f"Unsupported probs value type: {type(value)}")


def project_prob_vector_to_mer5(
    prob_vector: Sequence[float],
    native_label_names: Sequence[str],
    unsupported_policy: str = "drop",
    renormalize: bool = True,
) -> tuple[list[float], float]:
    if len(prob_vector) != len(native_label_names):
        raise ValueError(
            f"Length mismatch: len(prob_vector)={len(prob_vector)} "
            f"vs len(native_label_names)={len(native_label_names)}"
        )

    unsupported_policy = str(unsupported_policy).lower()
    if unsupported_policy not in {"drop", "map_to_neutral", "error"}:
        raise ValueError(f"Unsupported policy: {unsupported_policy}")

    projected = [0.0 for _ in MER5_LABELS]
    unsupported_mass = 0.0

    for native_label, prob in zip(native_label_names, prob_vector):
        target = UIT_VSMEC_NATIVE_TO_MER5.get(native_label)
        prob = float(prob)

        if target is None:
            unsupported_mass += prob
            if unsupported_policy == "map_to_neutral":
                projected[MER5_LABEL2ID["neutral"]] += prob
            elif unsupported_policy == "error" and prob > 0.0:
                raise ValueError(
                    f"Unsupported native label '{native_label}' has non-zero probability mass"
                )
            continue

        projected[MER5_LABEL2ID[target]] += prob

    if renormalize:
        total = sum(projected)
        if total > 0:
            projected = [p / total for p in projected]

    return projected, unsupported_mass


def project_native_label_to_mer5(label: str) -> str | None:
    return UIT_VSMEC_NATIVE_TO_MER5.get(label)


def project_prediction_dataframe_to_mer5(
    pred_df: pd.DataFrame,
    native_label_names: Sequence[str],
    unsupported_policy: str = "drop",
) -> pd.DataFrame:
    required_cols = {"sample_id", "label_id", "pred_id", "probs"}
    missing = required_cols - set(pred_df.columns)
    if missing:
        raise ValueError(f"Prediction dataframe missing required columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []

    for _, row in pred_df.iterrows():
        native_true_label = (
            str(row["label"])
            if "label" in pred_df.columns and pd.notna(row["label"])
            else native_label_names[int(row["label_id"])]
        )
        native_pred_label = (
            str(row["pred"])
            if "pred" in pred_df.columns and pd.notna(row["pred"])
            else native_label_names[int(row["pred_id"])]
        )

        prob_vector = _parse_probs(row["probs"])
        projected_probs, unsupported_mass = project_prob_vector_to_mer5(
            prob_vector=prob_vector,
            native_label_names=native_label_names,
            unsupported_policy=unsupported_policy,
            renormalize=True,
        )

        fusion_pred_id = int(max(range(len(projected_probs)), key=lambda i: projected_probs[i]))
        fusion_pred = MER5_LABELS[fusion_pred_id]
        fusion_true = project_native_label_to_mer5(native_true_label)

        out_row = dict(row)
        out_row["fusion_label"] = fusion_true
        out_row["fusion_label_id"] = (
            MER5_LABEL2ID[fusion_true] if fusion_true is not None else None
        )
        out_row["fusion_pred"] = fusion_pred
        out_row["fusion_pred_id"] = fusion_pred_id
        out_row["fusion_probs"] = json.dumps(projected_probs, ensure_ascii=False)
        out_row["unsupported_prob_mass"] = unsupported_mass
        out_row["native_true_label"] = native_true_label
        out_row["native_pred_label"] = native_pred_label
        out_row["is_supported_for_fusion_eval"] = fusion_true is not None
        rows.append(out_row)

    return pd.DataFrame(rows)


def compute_projected_mer5_metrics(projected_df: pd.DataFrame) -> dict[str, Any]:
    if "fusion_label_id" not in projected_df.columns or "fusion_pred_id" not in projected_df.columns:
        raise ValueError(
            "Projected dataframe must contain fusion_label_id and fusion_pred_id columns"
        )

    eval_df = projected_df[projected_df["fusion_label_id"].notna()].copy()

    if eval_df.empty:
        return {
            "num_rows": int(len(projected_df)),
            "num_supported_rows": 0,
            "num_dropped_rows": int(len(projected_df)),
            "avg_unsupported_prob_mass": float(projected_df["unsupported_prob_mass"].mean())
            if len(projected_df) > 0
            else 0.0,
            "metrics": {},
        }

    y_true = eval_df["fusion_label_id"].astype(int).tolist()
    y_pred = eval_df["fusion_pred_id"].astype(int).tolist()
    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=MER5_LABELS,
    )

    return {
        "num_rows": int(len(projected_df)),
        "num_supported_rows": int(len(eval_df)),
        "num_dropped_rows": int(len(projected_df) - len(eval_df)),
        "avg_unsupported_prob_mass": float(projected_df["unsupported_prob_mass"].mean())
        if len(projected_df) > 0
        else 0.0,
        "metrics": metrics,
    }


def save_projected_predictions(
    projected_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    projected_df.to_csv(file_path, index=False)