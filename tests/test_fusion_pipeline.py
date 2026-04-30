from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.fusion.confidence import ConfidenceFusionStrategy
from src.fusion.io import load_branch_predictions
from src.fusion.merger import merge_branch_predictions
from src.fusion.soft import SoftFusionStrategy
from src.fusion.threshold import ThresholdFusionStrategy


def _write_preds(path: Path, rows: list[dict]) -> None:
   df = pd.DataFrame(rows)
   df.to_csv(path, index=False)


def test_fusion_confidence_and_soft_pipeline(tmp_path: Path) -> None:
   text_csv = tmp_path / "text_valid.csv"
   speech_csv = tmp_path / "speech_valid.csv"

   common_rows = [
       {"sample_id": "a", "label_id": 0, "split": "valid"},
       {"sample_id": "b", "label_id": 1, "split": "valid"},
       {"sample_id": "c", "label_id": 0, "split": "valid"},
   ]

   _write_preds(
       text_csv,
       [
           {
               **common_rows[0],
               "pred_id": 0,
               "confidence": 0.90,
               "entropy": 0.20,
               "top2_margin": 0.80,
               "probs": "[0.9, 0.1]",
               "logits": "[2.2, 0.1]",
           },
           {
               **common_rows[1],
               "pred_id": 1,
               "confidence": 0.80,
               "entropy": 0.30,
               "top2_margin": 0.60,
               "probs": "[0.2, 0.8]",
               "logits": "[0.1, 2.0]",
           },
           {
               **common_rows[2],
               "pred_id": 1,
               "confidence": 0.55,
               "entropy": 0.68,
               "top2_margin": 0.10,
               "probs": "[0.45, 0.55]",
               "logits": "[0.3, 0.5]",
           },
       ],
   )
   _write_preds(
       speech_csv,
       [
           {
               **common_rows[0],
               "pred_id": 1,
               "confidence": 0.60,
               "entropy": 0.50,
               "top2_margin": 0.20,
               "probs": "[0.4, 0.6]",
               "logits": "[0.2, 0.8]",
           },
           {
               **common_rows[1],
               "pred_id": 0,
               "confidence": 0.51,
               "entropy": 0.69,
               "top2_margin": 0.02,
               "probs": "[0.51, 0.49]",
               "logits": "[0.52, 0.48]",
           },
           {
               **common_rows[2],
               "pred_id": 0,
               "confidence": 0.90,
               "entropy": 0.15,
               "top2_margin": 0.80,
               "probs": "[0.9, 0.1]",
               "logits": "[2.5, 0.1]",
           },
       ],
   )

   text_branch = load_branch_predictions(text_csv, modality="text")
   speech_branch = load_branch_predictions(speech_csv, modality="speech")
   merged = merge_branch_predictions(text_branch, speech_branch)

   confidence = ConfidenceFusionStrategy({"strategy": "confidence"})
   conf_bundle = confidence.predict(merged)
   assert conf_bundle.preds.tolist() == [0, 1, 0]

   soft = SoftFusionStrategy({"strategy": "soft", "soft": {"weight_grid": [0.0, 0.5, 1.0]}})
   soft.fit(merged)
   soft_bundle = soft.predict(merged)
   assert len(soft_bundle.sources) == 3
   assert soft_bundle.probs.shape == (3, 2)


def test_fusion_threshold_pipeline(tmp_path: Path) -> None:
   text_csv = tmp_path / "text_valid.csv"
   speech_csv = tmp_path / "speech_valid.csv"

   _write_preds(
       text_csv,
       [
           {
               "sample_id": "a",
               "label_id": 0,
               "pred_id": 0,
               "confidence": 0.95,
               "entropy": 0.10,
               "top2_margin": 0.90,
               "probs": "[0.95, 0.05]",
               "logits": "[2.9, 0.1]",
               "split": "valid",
           },
           {
               "sample_id": "b",
               "label_id": 1,
               "pred_id": 0,
               "confidence": 0.52,
               "entropy": 0.68,
               "top2_margin": 0.04,
               "probs": "[0.52, 0.48]",
               "logits": "[0.52, 0.48]",
               "split": "valid",
           },
       ],
   )
   _write_preds(
       speech_csv,
       [
           {
               "sample_id": "a",
               "label_id": 0,
               "pred_id": 1,
               "confidence": 0.51,
               "entropy": 0.69,
               "top2_margin": 0.02,
               "probs": "[0.49, 0.51]",
               "logits": "[0.49, 0.51]",
               "split": "valid",
           },
           {
               "sample_id": "b",
               "label_id": 1,
               "pred_id": 1,
               "confidence": 0.91,
               "entropy": 0.15,
               "top2_margin": 0.82,
               "probs": "[0.09, 0.91]",
               "logits": "[0.1, 2.5]",
               "split": "valid",
           },
       ],
   )

   merged = merge_branch_predictions(
       load_branch_predictions(text_csv, modality="text"),
       load_branch_predictions(speech_csv, modality="speech"),
   )

   strategy = ThresholdFusionStrategy(
       {"strategy": "threshold", "threshold": {"quantiles": [0.0, 0.5, 1.0]}}
   )
   strategy.fit(merged)
   bundle = strategy.predict(merged)
   assert bundle.preds.tolist() == [0, 1]
