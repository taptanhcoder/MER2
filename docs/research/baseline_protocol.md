# Baseline Freeze Protocol

This document defines the immutable baseline policy for the project.

## Goal

Before any further tuning:
- freeze the current text baseline
- freeze the current speech baseline
- freeze the current fusion baseline
- freeze the shared exported artifacts used by fusion

This ensures every later experiment is compared against one stable and reproducible anchor.

## Baseline V1

The following runs are treated as the official V1 baselines:

- Text: `outputs/runs/text/text_phobert_base_ce/seed_42`
- Speech: `outputs/runs/speech/speech_hubert_base_attn_freeze_ce/seed_42`
- Fusion: `outputs/runs/fusion/fusion_light_bica_gate_v2/seed_42`

## Rules

1. Do not edit or overwrite the configs that produced the baselines.
2. Do not reuse the same experiment names for new tuning runs.
3. Every new experiment must use a new experiment name.
4. Before tuning, create a frozen snapshot using `scripts/freeze_baseline.py`.
5. Treat the frozen snapshot as the single source of truth for baseline comparison.

## Required Snapshot Contents

For each baseline run:
- `resolved_config.yaml`
- `metrics.json`
- `best.ckpt`
- `classification_report.json`
- `confusion_matrix.json`
- `valid_predictions.csv`
- `test_predictions.csv`

Additional required artifacts:
- text tokenizer
- text/speech calibration outputs
- fusion branch metrics
- fusion dominance priors
- shared text/speech exports
- shared fusion artifacts

## Command

```bash
PYTHONPATH=. python scripts/freeze_baseline.py \
  --config configs/research/baseline_v1.yaml