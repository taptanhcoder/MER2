from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _to_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
   if isinstance(array, np.ndarray):
       return array
   return array.detach().cpu().numpy()


def softmax_numpy(logits: np.ndarray) -> np.ndarray:
   shifted = logits - logits.max(axis=1, keepdims=True)
   exp_values = np.exp(shifted)
   return exp_values / exp_values.sum(axis=1, keepdims=True)


def negative_log_likelihood_from_logits(
   logits: np.ndarray | torch.Tensor,
   labels: np.ndarray | torch.Tensor,
) -> float:
   logits_np = _to_numpy(logits).astype(np.float64)
   labels_np = _to_numpy(labels).astype(np.int64)

   probs = softmax_numpy(logits_np)
   eps = 1e-12
   true_probs = probs[np.arange(len(labels_np)), labels_np]
   nll = -np.log(np.clip(true_probs, eps, 1.0)).mean()
   return float(nll)


def brier_score_multiclass(
   probs: np.ndarray | torch.Tensor,
   labels: np.ndarray | torch.Tensor,
) -> float:
   probs_np = _to_numpy(probs).astype(np.float64)
   labels_np = _to_numpy(labels).astype(np.int64)

   one_hot = np.zeros_like(probs_np)
   one_hot[np.arange(len(labels_np)), labels_np] = 1.0
   score = ((probs_np - one_hot) ** 2).sum(axis=1).mean()
   return float(score)


def expected_calibration_error(
   probs: np.ndarray | torch.Tensor,
   labels: np.ndarray | torch.Tensor,
   n_bins: int = 15,
) -> float:
   probs_np = _to_numpy(probs).astype(np.float64)
   labels_np = _to_numpy(labels).astype(np.int64)

   confidences = probs_np.max(axis=1)
   predictions = probs_np.argmax(axis=1)
   correctness = (predictions == labels_np).astype(np.float64)

   bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
   ece = 0.0
   total = len(labels_np)

   for bin_idx in range(n_bins):
       left = bin_edges[bin_idx]
       right = bin_edges[bin_idx + 1]

       if bin_idx == n_bins - 1:
           in_bin = (confidences >= left) & (confidences <= right)
       else:
           in_bin = (confidences >= left) & (confidences < right)

       if not np.any(in_bin):
           continue

       bin_confidence = confidences[in_bin].mean()
       bin_accuracy = correctness[in_bin].mean()
       ece += (in_bin.sum() / total) * abs(bin_accuracy - bin_confidence)

   return float(ece)


def entropy_from_probs(probs: np.ndarray | torch.Tensor) -> np.ndarray:
   probs_np = _to_numpy(probs).astype(np.float64)
   eps = 1e-12
   entropy = -(probs_np * np.log(np.clip(probs_np, eps, 1.0))).sum(axis=1)
   return entropy.astype(np.float64)


def calibration_metrics_from_logits(
   logits: np.ndarray | torch.Tensor,
   labels: np.ndarray | torch.Tensor,
   n_bins: int = 15,
) -> dict[str, Any]:
   logits_np = _to_numpy(logits).astype(np.float64)
   labels_np = _to_numpy(labels).astype(np.int64)
   probs = softmax_numpy(logits_np)

   predictions = probs.argmax(axis=1)
   confidences = probs.max(axis=1)

   return {
       "nll": negative_log_likelihood_from_logits(logits_np, labels_np),
       "brier": brier_score_multiclass(probs, labels_np),
       "ece": expected_calibration_error(probs, labels_np, n_bins=n_bins),
       "accuracy": float((predictions == labels_np).mean()) if len(labels_np) > 0 else 0.0,
       "mean_confidence": float(confidences.mean()) if len(confidences) > 0 else 0.0,
       "mean_entropy": float(entropy_from_probs(probs).mean()) if len(probs) > 0 else 0.0,
   }
