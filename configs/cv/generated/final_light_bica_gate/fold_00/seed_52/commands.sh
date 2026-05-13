#!/usr/bin/env bash
set -euo pipefail

# Generated CV fold/seed run plan.
# Text/speech CV runs use fold/seed-specific experiment names.
# Fusion exports/artifacts/runs are isolated under outputs/cv/.

export PYTHONPATH=.

TEXT_RUN_DIR=outputs/runs/text/text_phobert_base_ce_headtail128_fold_00_seed_52/seed_52
SPEECH_RUN_DIR=outputs/runs/speech/speech_hubert_base_attn_freeze_ce_fold_00_seed_52/seed_52

# 1) Train text expert
if [ -f "$TEXT_RUN_DIR/resolved_config.yaml" ]; then
  echo "[SKIP] Existing text run: $TEXT_RUN_DIR"
else
  python scripts/train_text.py --config /home/emotalk/mer2/configs/cv/generated/final_light_bica_gate/fold_00/seed_52/text_headtail128.yaml
fi

# 2) Calibrate text expert
if [ -f "$TEXT_RUN_DIR/temperature.json" ]; then
  echo "[SKIP] Existing text calibration: $TEXT_RUN_DIR/temperature.json"
else
  python scripts/calibrate.py --run-dir "$TEXT_RUN_DIR"
fi

# 3) Export text features
python scripts/export_text_features.py --run-dir outputs/runs/text/text_phobert_base_ce_headtail128_fold_00_seed_52/seed_52 --output-dir /home/emotalk/mer2/outputs/cv/fusion_exports/text_headtail128/fold_00/seed_52 --device cuda

# 4) Train speech expert
if [ -f "$SPEECH_RUN_DIR/resolved_config.yaml" ]; then
  echo "[SKIP] Existing speech run: $SPEECH_RUN_DIR"
else
  python scripts/train_speech.py --config /home/emotalk/mer2/configs/cv/generated/final_light_bica_gate/fold_00/seed_52/speech_hubert_ce.yaml
fi

# 5) Calibrate speech expert
if [ -f "$SPEECH_RUN_DIR/temperature.json" ]; then
  echo "[SKIP] Existing speech calibration: $SPEECH_RUN_DIR/temperature.json"
else
  python scripts/calibrate.py --run-dir "$SPEECH_RUN_DIR"
fi

# 6) Export speech features
python scripts/export_speech_features.py --run-dir outputs/runs/speech/speech_hubert_base_attn_freeze_ce_fold_00_seed_52/seed_52 --output-dir /home/emotalk/mer2/outputs/cv/fusion_exports/speech_hubert_ce/fold_00/seed_52 --device cuda

# 7) Prepare fusion artifacts
python scripts/prepare_fusion_artifacts.py --config /home/emotalk/mer2/configs/cv/generated/final_light_bica_gate/fold_00/seed_52/prepare_fusion_artifacts.yaml

# 8) Train/evaluate final fusion
python scripts/run_fusion.py --config /home/emotalk/mer2/configs/cv/generated/final_light_bica_gate/fold_00/seed_52/fusion_final_light_bica_gate.yaml --seed 52

