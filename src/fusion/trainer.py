from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.common.base_trainer import BaseTrainer
from src.common.interfaces import ClassifierOutput, EpochResult
from src.fusion.dataset import build_fusion_dataloaders
from src.fusion.losses import dominance_regularizer


def _entropy(prob_row: list[float]) -> float:
    eps = 1e-12
    return float(-sum(p * math.log(max(p, eps)) for p in prob_row))


def _top2_margin(prob_row: list[float]) -> float:
    if not prob_row:
        return 0.0
    top2 = sorted(prob_row, reverse=True)[:2]
    if len(top2) == 1:
        return float(top2[0])
    return float(top2[0] - top2[1])


def _confidence_from_probs_tensor(probs: torch.Tensor) -> list[float]:
    return probs.max(dim=-1).values.detach().cpu().tolist()


class FusionTrainer(BaseTrainer):
    def __init__(
        self,
        *args,
        prior_text: torch.Tensor,
        prior_speech: torch.Tensor,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.prior_text = prior_text.detach().float().to(self.device)
        self.prior_speech = prior_speech.detach().float().to(self.device)
        self._last_aux: dict[str, torch.Tensor] | None = None

    def _validate_batch_contract(self, batch: dict[str, Any]) -> None:
        ids = [str(x) for x in batch["ids"]]

        for key in [
            "text_tokens",
            "speech_tokens",
            "text_embedding",
            "speech_embedding",
            "text_logits_raw",
            "speech_logits_raw",
            "text_probs_cal",
            "speech_probs_cal",
            "reliability",
        ]:
            tensor = batch[key]
            if not torch.isfinite(tensor).all():
                bad_idx = (~torch.isfinite(tensor.view(tensor.shape[0], -1))).any(dim=1).nonzero(as_tuple=False).view(-1)
                examples = [ids[i] for i in bad_idx[:5].tolist()]
                raise ValueError(f"Batch tensor {key} contains NaN/Inf. Example sample_ids: {examples}")

        for key in ["text_token_mask", "speech_token_mask"]:
            mask = batch[key]
            if mask.dim() != 2:
                raise ValueError(f"Batch mask {key} must be 2D, got {tuple(mask.shape)}")
            invalid = (mask.sum(dim=1) <= 0).nonzero(as_tuple=False).view(-1)
            if invalid.numel() > 0:
                examples = [ids[i] for i in invalid[:5].tolist()]
                raise ValueError(f"Batch mask {key} has zero-valid-token samples. Examples: {examples}")

    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        self._validate_batch_contract(batch)

        return {
            "ids": batch["ids"],
            "labels": batch["labels"].to(self.device),
            "text_logits_raw": batch["text_logits_raw"].to(self.device),
            "speech_logits_raw": batch["speech_logits_raw"].to(self.device),
            "text_probs_cal": batch["text_probs_cal"].to(self.device),
            "speech_probs_cal": batch["speech_probs_cal"].to(self.device),
            "text_embedding": batch["text_embedding"].to(self.device),
            "speech_embedding": batch["speech_embedding"].to(self.device),
            "text_tokens": batch["text_tokens"].to(self.device),
            "speech_tokens": batch["speech_tokens"].to(self.device),
            "text_token_mask": batch["text_token_mask"].to(self.device),
            "speech_token_mask": batch["speech_token_mask"].to(self.device),
            "reliability": batch["reliability"].to(self.device),
            "prediction_metadata": {
                "texts": [str(x) for x in batch.get("texts", [])],
                "raw_texts": [str(x) for x in batch.get("raw_texts", [])],
                "audio_paths": [str(x) for x in batch.get("audio_paths", [])],
                "group_ids": [str(x) for x in batch.get("group_ids", [])],
                "durations": [float(x) for x in batch.get("durations", [])],
            },
        }

    def forward_step(self, batch: dict[str, Any]) -> ClassifierOutput:
        output, aux = self.model(
            text_embedding=batch["text_embedding"],
            speech_embedding=batch["speech_embedding"],
            text_tokens=batch["text_tokens"],
            speech_tokens=batch["speech_tokens"],
            text_token_mask=batch["text_token_mask"],
            speech_token_mask=batch["speech_token_mask"],
            text_logits_raw=batch["text_logits_raw"],
            speech_logits_raw=batch["speech_logits_raw"],
            text_probs_cal=batch["text_probs_cal"],
            speech_probs_cal=batch["speech_probs_cal"],
            reliability=batch["reliability"],
            labels=batch["labels"],
            return_aux=True,
        )
        self._last_aux = aux

        prediction_metadata = batch.setdefault("prediction_metadata", {})
        prediction_metadata["alpha_text_mean"] = aux["alpha_text"].mean(dim=-1).detach().cpu().tolist()
        prediction_metadata["alpha_speech_mean"] = aux["alpha_speech"].mean(dim=-1).detach().cpu().tolist()
        prediction_metadata["alpha_interaction_mean"] = aux["alpha_interaction"].mean(dim=-1).detach().cpu().tolist()

        prediction_metadata["text_pred_id"] = aux["text_preds_raw"].detach().cpu().tolist()
        prediction_metadata["speech_pred_id"] = aux["speech_preds_raw"].detach().cpu().tolist()
        prediction_metadata["interaction_pred_id"] = aux["interaction_preds"].detach().cpu().tolist()

        prediction_metadata["text_confidence"] = _confidence_from_probs_tensor(aux["text_probs_raw"])
        prediction_metadata["speech_confidence"] = _confidence_from_probs_tensor(aux["speech_probs_raw"])
        prediction_metadata["interaction_confidence"] = _confidence_from_probs_tensor(aux["interaction_probs"])

        return output

    def compute_loss(self, output: ClassifierOutput, batch: dict[str, Any]) -> torch.Tensor:
        ce_loss = super().compute_loss(output, batch)

        dominance_cfg = dict(self.config.get("dominance", {}))
        if not bool(dominance_cfg.get("use_dominance_regularizer", False)):
            return ce_loss
        if self._last_aux is None:
            return ce_loss

        dom_loss = dominance_regularizer(
            alpha_text=self._last_aux["alpha_text"],
            alpha_speech=self._last_aux["alpha_speech"],
            prior_text=self.prior_text,
            prior_speech=self.prior_speech,
            lambda_text=float(dominance_cfg.get("lambda_text", 1.0)),
            lambda_speech=float(dominance_cfg.get("lambda_speech", 1.0)),
        )
        return ce_loss + dom_loss


def epoch_result_to_dataframe(
    result: EpochResult,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    texts = result.predictions.metadata.get("texts", [])
    raw_texts = result.predictions.metadata.get("raw_texts", [])
    audio_paths = result.predictions.metadata.get("audio_paths", [])
    group_ids = result.predictions.metadata.get("group_ids", [])
    durations = result.predictions.metadata.get("durations", [])
    alpha_text_mean = result.predictions.metadata.get("alpha_text_mean", [])
    alpha_speech_mean = result.predictions.metadata.get("alpha_speech_mean", [])
    alpha_interaction_mean = result.predictions.metadata.get("alpha_interaction_mean", [])

    text_pred_id = result.predictions.metadata.get("text_pred_id", [])
    speech_pred_id = result.predictions.metadata.get("speech_pred_id", [])
    interaction_pred_id = result.predictions.metadata.get("interaction_pred_id", [])

    text_confidence = result.predictions.metadata.get("text_confidence", [])
    speech_confidence = result.predictions.metadata.get("speech_confidence", [])
    interaction_confidence = result.predictions.metadata.get("interaction_confidence", [])

    for idx, (sample_id, label_id, pred_id, prob_row, logit_row) in enumerate(
        zip(
            result.predictions.ids,
            result.predictions.labels,
            result.predictions.preds,
            result.predictions.probs,
            result.predictions.logits,
        )
    ):
        row = {
            "sample_id": sample_id,
            "label_id": int(label_id),
            "pred_id": int(pred_id),
            "confidence": float(max(prob_row)),
            "entropy": _entropy(prob_row),
            "top2_margin": _top2_margin(prob_row),
            "probs": json.dumps(prob_row, ensure_ascii=False),
            "logits": json.dumps(logit_row, ensure_ascii=False),
            "split": result.split,
        }

        if idx < len(texts):
            row["text"] = texts[idx]
        if idx < len(raw_texts):
            row["raw_text"] = raw_texts[idx]
        if idx < len(audio_paths):
            row["audio_path"] = audio_paths[idx]
        if idx < len(group_ids):
            row["group_id"] = group_ids[idx]
        if idx < len(durations):
            row["duration"] = float(durations[idx])

        if idx < len(alpha_text_mean):
            row["alpha_text_mean"] = float(alpha_text_mean[idx])
        if idx < len(alpha_speech_mean):
            row["alpha_speech_mean"] = float(alpha_speech_mean[idx])
        if idx < len(alpha_interaction_mean):
            row["alpha_interaction_mean"] = float(alpha_interaction_mean[idx])

        if idx < len(text_pred_id):
            row["text_pred_id"] = int(text_pred_id[idx])
        if idx < len(speech_pred_id):
            row["speech_pred_id"] = int(speech_pred_id[idx])
        if idx < len(interaction_pred_id):
            row["interaction_pred_id"] = int(interaction_pred_id[idx])

        if idx < len(text_confidence):
            row["text_confidence"] = float(text_confidence[idx])
        if idx < len(speech_confidence):
            row["speech_confidence"] = float(speech_confidence[idx])
        if idx < len(interaction_confidence):
            row["interaction_confidence"] = float(interaction_confidence[idx])

        if label_names is not None:
            row["label"] = label_names[int(label_id)]
            row["pred"] = label_names[int(pred_id)]

            if "text_pred_id" in row:
                row["text_pred"] = label_names[int(row["text_pred_id"])]
            if "speech_pred_id" in row:
                row["speech_pred"] = label_names[int(row["speech_pred_id"])]
            if "interaction_pred_id" in row:
                row["interaction_pred"] = label_names[int(row["interaction_pred_id"])]

        rows.append(row)

    return pd.DataFrame(rows)


def save_epoch_predictions(
    result: EpochResult,
    output_path: str | Path,
    label_names: list[str] | None = None,
) -> None:
    df = epoch_result_to_dataframe(result, label_names=label_names)
    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)


__all__ = [
    "FusionTrainer",
    "build_fusion_dataloaders",
    "epoch_result_to_dataframe",
    "save_epoch_predictions",
]