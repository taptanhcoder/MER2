#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=${PYTHONPATH:-.}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false

TEXT_RUN_ROOT=/home/emotalk/mer2/outputs/meld/runs/text
SPEECH_RUN_ROOT=/home/emotalk/mer2/outputs/meld/runs/speech
TEXT_RUN_DIR=/home/emotalk/mer2/outputs/meld/runs/text/text_phobert_base_ce_headtail128/seed_42
SPEECH_RUN_DIR=/home/emotalk/mer2/outputs/meld/runs/speech/speech_hubert_base_attn_freeze_ce/seed_42
TEXT_EXPORT_DIR=/home/emotalk/mer2/outputs/meld/fusion_exports/text_headtail128/seed_42
SPEECH_EXPORT_DIR=/home/emotalk/mer2/outputs/meld/fusion_exports/speech_hubert/seed_42
FUSION_ARTIFACT_DIR=/home/emotalk/mer2/outputs/meld/fusion_artifacts/light_bica_gate/seed_42
FUSION_RUN_DIR=/home/emotalk/mer2/outputs/meld/runs/fusion/fusion_light_bica_gate_text_headtail128_cal_gateprob_v1/seed_42

echo "[MELD] Train text expert"
if [ -f "$TEXT_RUN_DIR/metrics.json" ]; then
  echo "[SKIP] Existing text run: $TEXT_RUN_DIR"
else
  python scripts/train_text.py --config /home/emotalk/mer2/configs/meld/generated/meld_light_bica_gate/seed_42/text_phobert_headtail128.yaml --seed 42 --output-root "$TEXT_RUN_ROOT"
fi

echo "[MELD] Calibrate text expert"
if [ ! -d "$TEXT_RUN_DIR" ]; then
  echo "[ERROR] Expected text run dir not found: $TEXT_RUN_DIR"
  echo "[ERROR] train_text.py likely wrote to the wrong output-root."
  exit 1
fi
if [ -f "$TEXT_RUN_DIR/temperature.json" ]; then
  echo "[SKIP] Existing text calibration: $TEXT_RUN_DIR/temperature.json"
else
  python scripts/calibrate.py --run-dir "$TEXT_RUN_DIR"
fi

echo "[MELD] Export text features"
if [ -f "$TEXT_EXPORT_DIR/train.pt" ] && [ -f "$TEXT_EXPORT_DIR/valid.pt" ] && [ -f "$TEXT_EXPORT_DIR/test.pt" ]; then
  echo "[SKIP] Existing text exports: $TEXT_EXPORT_DIR"
else
  python scripts/export_text_features.py --run-dir "$TEXT_RUN_DIR" --output-dir "$TEXT_EXPORT_DIR"
fi

echo "[MELD] Train speech expert"
if [ -f "$SPEECH_RUN_DIR/metrics.json" ]; then
  echo "[SKIP] Existing speech run: $SPEECH_RUN_DIR"
else
  python scripts/train_speech.py --config /home/emotalk/mer2/configs/meld/generated/meld_light_bica_gate/seed_42/speech_hubert.yaml --seed 42 --output-root "$SPEECH_RUN_ROOT"
fi

echo "[MELD] Calibrate speech expert"
if [ ! -d "$SPEECH_RUN_DIR" ]; then
  echo "[ERROR] Expected speech run dir not found: $SPEECH_RUN_DIR"
  echo "[ERROR] train_speech.py likely wrote to the wrong output-root."
  exit 1
fi
if [ -f "$SPEECH_RUN_DIR/temperature.json" ]; then
  echo "[SKIP] Existing speech calibration: $SPEECH_RUN_DIR/temperature.json"
else
  python scripts/calibrate.py --run-dir "$SPEECH_RUN_DIR"
fi

echo "[MELD] Export speech features"
if [ -f "$SPEECH_EXPORT_DIR/train.pt" ] && [ -f "$SPEECH_EXPORT_DIR/valid.pt" ] && [ -f "$SPEECH_EXPORT_DIR/test.pt" ]; then
  echo "[SKIP] Existing speech exports: $SPEECH_EXPORT_DIR"
else
  python scripts/export_speech_features.py --run-dir "$SPEECH_RUN_DIR" --output-dir "$SPEECH_EXPORT_DIR"
fi

echo "[MELD] Prepare fusion artifacts"
if [ -f "$FUSION_ARTIFACT_DIR/train.pt" ] && [ -f "$FUSION_ARTIFACT_DIR/valid.pt" ] && [ -f "$FUSION_ARTIFACT_DIR/test.pt" ]; then
  echo "[SKIP] Existing fusion artifacts: $FUSION_ARTIFACT_DIR"
else
  python scripts/prepare_fusion_artifacts.py --config /home/emotalk/mer2/configs/meld/generated/meld_light_bica_gate/seed_42/prepare_fusion_artifacts.yaml
fi

echo "[MELD] Train fusion model"
if [ -f "$FUSION_RUN_DIR/metrics.json" ]; then
  echo "[SKIP] Existing fusion run: $FUSION_RUN_DIR"
else
  python scripts/run_fusion.py --config /home/emotalk/mer2/configs/meld/generated/meld_light_bica_gate/seed_42/fusion_light_bica_gate.yaml
fi

echo "[OK] MELD framework evaluation finished."
