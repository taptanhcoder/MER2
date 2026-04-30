from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.common.interfaces import ClassifierOutput
from src.common.registry import MODEL_REGISTRY
from src.fusion.gates import ClassWiseAdaptiveGate
from src.fusion.interaction import BiDirectionalCrossAttention


def _resolve_num_classes(config: dict[str, Any]) -> int:
    dataset_cfg = dict(config.get("dataset", {}))
    label_space_cfg = dict(config.get("label_space", {}))
    model_cfg = dict(config.get("model", {}))

    if model_cfg.get("num_classes") is not None:
        return int(model_cfg["num_classes"])
    if dataset_cfg.get("num_classes") is not None:
        return int(dataset_cfg["num_classes"])
    labels = label_space_cfg.get("labels")
    if labels is not None:
        return int(len(labels))
    raise KeyError("Could not resolve fusion num_classes.")


@MODEL_REGISTRY.register("late-logit-avg")
@MODEL_REGISTRY.register("calibrated-two-way-gate")
@MODEL_REGISTRY.register("light-bica-gate")
class FusionClassifier(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        model_cfg = dict(config.get("model", {}))
        dominance_cfg = dict(config.get("dominance", {}))

        self.num_classes = _resolve_num_classes(config)
        self.text_token_dim = int(model_cfg["text_token_dim"])
        self.speech_token_dim = int(model_cfg["speech_token_dim"])
        self.text_embedding_dim = int(model_cfg["text_embedding_dim"])
        self.speech_embedding_dim = int(model_cfg["speech_embedding_dim"])
        self.fusion_dim = int(model_cfg.get("fusion_dim", 128))
        self.use_interaction = bool(model_cfg.get("use_interaction", True))
        self.gate_mode = str(model_cfg.get("gate_mode", "three_way_classwise"))
        self.use_residual_logit_correction = bool(
            model_cfg.get("use_residual_logit_correction", False)
        )

        self.use_modality_dropout = bool(dominance_cfg.get("use_modality_dropout", False))
        self.text_dropout_prob = float(dominance_cfg.get("text_dropout_prob", 0.0))
        self.speech_dropout_prob = float(dominance_cfg.get("speech_dropout_prob", 0.0))

        self.interaction_block = BiDirectionalCrossAttention(
            text_dim=self.text_token_dim,
            speech_dim=self.speech_token_dim,
            fusion_dim=self.fusion_dim,
            dropout=float(model_cfg.get("interaction_dropout", 0.1)),
        )
        self.text_embedding_proj = nn.Linear(self.text_embedding_dim, self.fusion_dim)
        self.speech_embedding_proj = nn.Linear(self.speech_embedding_dim, self.fusion_dim)
        self.interaction_embed_norm = nn.LayerNorm(self.fusion_dim)

        hidden_dim = int(model_cfg.get("interaction_hidden_dim", 256))
        dropout = float(model_cfg.get("interaction_dropout", 0.1))
        self.interaction_head = nn.Sequential(
            nn.Linear(self.fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

        reliability_dim = int(model_cfg["reliability_dim"])
        gate_input_dim = (self.num_classes * 3) + reliability_dim

        self.gate = ClassWiseAdaptiveGate(
            input_dim=gate_input_dim,
            num_classes=self.num_classes,
            mode=self.gate_mode,
            hidden_dim=int(model_cfg.get("gate_hidden_dim", 128)),
            dropout=float(model_cfg.get("gate_dropout", 0.1)),
        )

        self.residual = None
        if self.use_residual_logit_correction:
            self.residual = nn.Linear(gate_input_dim, self.num_classes)

    def _validate_fusion_contract(
        self,
        text_embedding: torch.Tensor,
        speech_embedding: torch.Tensor,
        text_tokens: torch.Tensor,
        speech_tokens: torch.Tensor,
        text_token_mask: torch.Tensor,
        speech_token_mask: torch.Tensor,
        text_logits_raw: torch.Tensor,
        speech_logits_raw: torch.Tensor,
        text_probs_cal: torch.Tensor,
        speech_probs_cal: torch.Tensor,
        reliability: torch.Tensor,
    ) -> None:
        if text_tokens.dim() != 3 or speech_tokens.dim() != 3:
            raise ValueError(
                f"Fusion token banks must be 3D. "
                f"Got text_tokens={tuple(text_tokens.shape)}, speech_tokens={tuple(speech_tokens.shape)}"
            )
        if text_token_mask.dim() != 2 or speech_token_mask.dim() != 2:
            raise ValueError(
                f"Fusion token masks must be 2D. "
                f"Got text_token_mask={tuple(text_token_mask.shape)}, speech_token_mask={tuple(speech_token_mask.shape)}"
            )

        batch_size = int(text_tokens.shape[0])
        expected_first_dims = {
            "speech_tokens": int(speech_tokens.shape[0]),
            "text_embedding": int(text_embedding.shape[0]),
            "speech_embedding": int(speech_embedding.shape[0]),
            "text_logits_raw": int(text_logits_raw.shape[0]),
            "speech_logits_raw": int(speech_logits_raw.shape[0]),
            "text_probs_cal": int(text_probs_cal.shape[0]),
            "speech_probs_cal": int(speech_probs_cal.shape[0]),
            "reliability": int(reliability.shape[0]),
            "text_token_mask": int(text_token_mask.shape[0]),
            "speech_token_mask": int(speech_token_mask.shape[0]),
        }
        mismatched = {k: v for k, v in expected_first_dims.items() if v != batch_size}
        if mismatched:
            raise ValueError(
                f"Fusion batch first-dimension mismatch. "
                f"Expected batch_size={batch_size}, got {mismatched}"
            )

        if text_tokens.shape[:2] != text_token_mask.shape:
            raise ValueError(
                f"text_tokens/text_token_mask shape mismatch: "
                f"{tuple(text_tokens.shape)} vs {tuple(text_token_mask.shape)}"
            )
        if speech_tokens.shape[:2] != speech_token_mask.shape:
            raise ValueError(
                f"speech_tokens/speech_token_mask shape mismatch: "
                f"{tuple(speech_tokens.shape)} vs {tuple(speech_token_mask.shape)}"
            )

        for name, tensor in {
            "text_embedding": text_embedding,
            "speech_embedding": speech_embedding,
            "text_tokens": text_tokens,
            "speech_tokens": speech_tokens,
            "text_logits_raw": text_logits_raw,
            "speech_logits_raw": speech_logits_raw,
            "text_probs_cal": text_probs_cal,
            "speech_probs_cal": speech_probs_cal,
            "reliability": reliability,
        }.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Fusion input {name} contains NaN/Inf.")

        if (text_token_mask.sum(dim=1) <= 0).any():
            raise ValueError("Fusion input text_token_mask has sample(s) with zero valid tokens.")
        if (speech_token_mask.sum(dim=1) <= 0).any():
            raise ValueError("Fusion input speech_token_mask has sample(s) with zero valid tokens.")

    def _apply_modality_dropout(
        self,
        text_tokens: torch.Tensor,
        speech_tokens: torch.Tensor,
        text_logits: torch.Tensor,
        speech_logits: torch.Tensor,
        text_probs: torch.Tensor,
        speech_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.training or not self.use_modality_dropout:
            return text_tokens, speech_tokens, text_logits, speech_logits, text_probs, speech_probs

        batch_size = int(text_logits.shape[0])
        device = text_logits.device

        drop_text = torch.rand(batch_size, device=device) < self.text_dropout_prob
        drop_speech = torch.rand(batch_size, device=device) < self.speech_dropout_prob

        both = torch.logical_and(drop_text, drop_speech)
        drop_speech = torch.where(both, torch.zeros_like(drop_speech), drop_speech)

        keep_text = (~drop_text).float().unsqueeze(-1)
        keep_speech = (~drop_speech).float().unsqueeze(-1)

        text_tokens = text_tokens * keep_text.unsqueeze(-1)
        speech_tokens = speech_tokens * keep_speech.unsqueeze(-1)

        text_logits = text_logits * keep_text
        speech_logits = speech_logits * keep_speech
        text_probs = text_probs * keep_text
        speech_probs = speech_probs * keep_speech
        return text_tokens, speech_tokens, text_logits, speech_logits, text_probs, speech_probs

    def forward(
        self,
        text_embedding: torch.Tensor,
        speech_embedding: torch.Tensor,
        text_tokens: torch.Tensor,
        speech_tokens: torch.Tensor,
        text_token_mask: torch.Tensor,
        speech_token_mask: torch.Tensor,
        text_logits_raw: torch.Tensor,
        speech_logits_raw: torch.Tensor,
        text_probs_cal: torch.Tensor,
        speech_probs_cal: torch.Tensor,
        reliability: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_aux: bool = False,
        **kwargs,
    ) -> ClassifierOutput | tuple[ClassifierOutput, dict[str, torch.Tensor]]:
        self._validate_fusion_contract(
            text_embedding=text_embedding,
            speech_embedding=speech_embedding,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
            text_token_mask=text_token_mask,
            speech_token_mask=speech_token_mask,
            text_logits_raw=text_logits_raw,
            speech_logits_raw=speech_logits_raw,
            text_probs_cal=text_probs_cal,
            speech_probs_cal=speech_probs_cal,
            reliability=reliability,
        )

        (
            text_tokens,
            speech_tokens,
            text_logits_raw,
            speech_logits_raw,
            text_probs_cal,
            speech_probs_cal,
        ) = self._apply_modality_dropout(
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
            text_logits=text_logits_raw,
            speech_logits=speech_logits_raw,
            text_probs=text_probs_cal,
            speech_probs=speech_probs_cal,
        )

        text_probs_raw = torch.softmax(text_logits_raw, dim=-1)
        speech_probs_raw = torch.softmax(speech_logits_raw, dim=-1)
        text_preds_raw = torch.argmax(text_probs_raw, dim=-1)
        speech_preds_raw = torch.argmax(speech_probs_raw, dim=-1)

        if self.use_interaction:
            interaction_embedding, _ = self.interaction_block(
                text_tokens=text_tokens,
                text_mask=text_token_mask,
                speech_tokens=speech_tokens,
                speech_mask=speech_token_mask,
            )

            embedding_residual = 0.5 * (
                self.text_embedding_proj(text_embedding) + self.speech_embedding_proj(speech_embedding)
            )
            interaction_embedding = self.interaction_embed_norm(interaction_embedding + embedding_residual)
            interaction_logits = self.interaction_head(interaction_embedding)
        else:
            interaction_embedding = torch.zeros(
                text_logits_raw.shape[0],
                self.fusion_dim,
                dtype=text_logits_raw.dtype,
                device=text_logits_raw.device,
            )
            interaction_logits = torch.zeros_like(text_logits_raw)

        interaction_probs = torch.softmax(interaction_logits, dim=-1)
        interaction_preds = torch.argmax(interaction_probs, dim=-1)

        gate_input = torch.cat(
            [
                text_probs_cal,
                speech_probs_cal,
                interaction_logits,
                reliability,
            ],
            dim=-1,
        )

        alpha_text, alpha_speech, alpha_interaction = self.gate(gate_input)

        fused_logits = (
            alpha_text * text_logits_raw
            + alpha_speech * speech_logits_raw
            + alpha_interaction * interaction_logits
        )

        if self.residual is not None:
            fused_logits = fused_logits + self.residual(gate_input)

        probs = torch.softmax(fused_logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        output = ClassifierOutput(
            loss=None,
            logits=fused_logits,
            probs=probs,
            preds=preds,
            labels=labels,
            embeddings=interaction_embedding,
        )

        if not return_aux:
            return output

        aux = {
            "alpha_text": alpha_text,
            "alpha_speech": alpha_speech,
            "alpha_interaction": alpha_interaction,
            "text_logits_raw": text_logits_raw,
            "speech_logits_raw": speech_logits_raw,
            "interaction_logits": interaction_logits,
            "text_probs_raw": text_probs_raw,
            "speech_probs_raw": speech_probs_raw,
            "interaction_probs": interaction_probs,
            "text_preds_raw": text_preds_raw,
            "speech_preds_raw": speech_preds_raw,
            "interaction_preds": interaction_preds,
        }
        return output, aux