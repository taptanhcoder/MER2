from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
   sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration.io import (
   build_calibrated_prediction_frame,
   extract_logits_and_labels,
   load_prediction_frame,
)
from src.calibration.metrics import calibration_metrics_from_logits
from src.calibration.temperature_scaling import TemperatureScaler
from src.utils.config import deep_update, load_yaml
from src.utils.io import write_json, write_yaml
from src.utils.paths import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
   parser = argparse.ArgumentParser(
       description="Fit temperature scaling on validation predictions and apply to valid/test."
   )
   parser.add_argument(
       "--run-dir",
       type=Path,
       required=True,
       help="Run directory containing valid_predictions.csv and test_predictions.csv.",
   )
   parser.add_argument(
       "--config",
       type=Path,
       default=Path("configs/calibration/temperature_scaling.yaml"),
       help="Calibration config path.",
   )
   parser.add_argument(
       "--valid-predictions",
       type=Path,
       default=None,
       help="Optional override path for valid predictions CSV.",
   )
   parser.add_argument(
       "--test-predictions",
       type=Path,
       default=None,
       help="Optional override path for test predictions CSV.",
   )
   return parser


def resolve_calibration_config(config_path: Path) -> dict[str, Any]:
   cfg_path = resolve_project_path(config_path, start=PROJECT_ROOT)
   exp_cfg = load_yaml(cfg_path)
   defaults = exp_cfg.get("defaults", [])

   merged: dict[str, Any] = {}
   for entry in defaults:
       if not isinstance(entry, str):
           raise TypeError("Calibration defaults entries must be string paths.")
       part_cfg = load_yaml(resolve_project_path(entry, start=PROJECT_ROOT))
       merged = deep_update(merged, part_cfg)

   exp_without_defaults = dict(exp_cfg)
   exp_without_defaults.pop("defaults", None)
   merged = deep_update(merged, exp_without_defaults)
   return merged


def main() -> int:
   args = build_parser().parse_args()

   run_dir = resolve_project_path(args.run_dir, start=PROJECT_ROOT)
   if not run_dir.exists():
       raise FileNotFoundError(f"Run dir not found: {run_dir}")

   config = resolve_calibration_config(args.config)
   calibration_cfg = config["calibration"]
   output_cfg = config.get("output", {})

   valid_csv = (
       resolve_project_path(args.valid_predictions, start=PROJECT_ROOT)
       if args.valid_predictions is not None
       else run_dir / "valid_predictions.csv"
   )
   test_csv = (
       resolve_project_path(args.test_predictions, start=PROJECT_ROOT)
       if args.test_predictions is not None
       else run_dir / "test_predictions.csv"
   )

   valid_frame = load_prediction_frame(valid_csv)
   test_frame = load_prediction_frame(test_csv)

   valid_logits, valid_labels = extract_logits_and_labels(valid_frame)
   test_logits, test_labels = extract_logits_and_labels(test_frame)

   before_metrics = {
       "valid": calibration_metrics_from_logits(
           valid_logits,
           valid_labels,
           n_bins=int(calibration_cfg.get("n_bins", 15)),
       ),
       "test": calibration_metrics_from_logits(
           test_logits,
           test_labels,
           n_bins=int(calibration_cfg.get("n_bins", 15)),
       ),
   }

   method_name = str(calibration_cfg.get("method", "temperature_scaling")).lower()
   if method_name != "temperature_scaling":
       raise ValueError(
           f"Unsupported calibration method: {method_name}. Only temperature_scaling is implemented."
       )

   calibrator = TemperatureScaler(
       init_temperature=float(calibration_cfg.get("init_temperature", 1.0)),
       min_temperature=float(calibration_cfg.get("min_temperature", 0.05)),
       max_temperature=float(calibration_cfg.get("max_temperature", 10.0)),
       optimizer_config=dict(calibration_cfg.get("optimizer", {})),
   )
   calibrator.fit(valid_logits, valid_labels)

   calibrated_valid_logits = calibrator.transform_logits(valid_logits)
   calibrated_test_logits = calibrator.transform_logits(test_logits)

   after_metrics = {
       "valid": calibration_metrics_from_logits(
           calibrated_valid_logits,
           valid_labels,
           n_bins=int(calibration_cfg.get("n_bins", 15)),
       ),
       "test": calibration_metrics_from_logits(
           calibrated_test_logits,
           test_labels,
           n_bins=int(calibration_cfg.get("n_bins", 15)),
       ),
   }

   valid_calibrated_frame = build_calibrated_prediction_frame(
       original_frame=valid_frame,
       calibrated_logits=calibrated_valid_logits,
       method_name=method_name,
       temperature=calibrator.temperature,
   )
   test_calibrated_frame = build_calibrated_prediction_frame(
       original_frame=test_frame,
       calibrated_logits=calibrated_test_logits,
       method_name=method_name,
       temperature=calibrator.temperature,
   )

   if bool(output_cfg.get("save_calibrated_predictions", True)):
       valid_calibrated_frame.to_csv(
           run_dir / "valid_predictions_calibrated.csv",
           index=False,
       )
       test_calibrated_frame.to_csv(
           run_dir / "test_predictions_calibrated.csv",
           index=False,
       )

   temperature_payload = calibrator.state_dict()
   calibration_metrics_payload = {
       "method": method_name,
       "before": before_metrics,
       "after": after_metrics,
   }

   write_json(temperature_payload, run_dir / "temperature.json")
   write_json(calibration_metrics_payload, run_dir / "calibration_metrics.json")
   write_yaml(config, run_dir / "resolved_calibration_config.yaml")

   print(f"[OK] Calibrated run: {run_dir}")
   print(json.dumps(calibration_metrics_payload, ensure_ascii=False, indent=2))
   return 0


if __name__ == "__main__":
   raise SystemExit(main())
