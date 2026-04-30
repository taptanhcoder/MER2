from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModel

from src.common.interfaces import ClassifierOutput
from src.common.registry import MODEL_REGISTRY
from src.text.heads import ClassificationHead
from src.text.models.base import BaseTextClassifier


def _extract_model_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "model" in config:
        return dict(config["model"]), dict(config.get("finetune", {}))
    return dict(config), {}


def _get_encoder_layers(encoder: torch.nn.Module):
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return encoder.encoder.layer
    if hasattr(encoder, "layer"):
        return encoder.layer
    return None


@MODEL_REGISTRY.register("phobert-base")
@MODEL_REGISTRY.register("phobert-base-v2")
class PhoBERTClassifier(BaseTextClassifier):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.model_config, self.finetune_config = _extract_model_config(config)

        pretrained_name = self.model_config["pretrained_name"]
        self.pooling = str(self.model_config.get("pooling", "cls")).lower()
        self.num_classes = int(self.model_config["num_classes"])
        self.dropout = float(self.model_config.get("dropout", 0.1))

        hf_config = AutoConfig.from_pretrained(pretrained_name)
        self.encoder = AutoModel.from_pretrained(pretrained_name, config=hf_config)

        if bool(self.model_config.get("gradient_checkpointing", False)) and hasattr(
            self.encoder, "gradient_checkpointing_enable"
        ):
            self.encoder.gradient_checkpointing_enable()

        hidden_size = int(getattr(hf_config, "hidden_size"))
        self.classifier = ClassificationHead(
            input_dim=hidden_size,
            num_classes=self.num_classes,
            dropout=float(self.model_config.get("classifier_dropout", self.dropout)),
            hidden_dim=self.model_config.get("classifier_hidden_dim"),
            activation=str(self.model_config.get("classifier_activation", "gelu")),
            use_layer_norm=bool(self.model_config.get("classifier_layer_norm", True)),
        )

        self._apply_finetune_policy()
        self.encoder_trainable = any(param.requires_grad for param in self.encoder.parameters())

    def _freeze_module(self, module: torch.nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = False

    def _unfreeze_module(self, module: torch.nn.Module) -> None:
        for param in module.parameters():
            param.requires_grad = True

    def _apply_finetune_policy(self) -> None:
        freeze_embeddings = bool(self.finetune_config.get("freeze_embeddings", False))
        train_full_encoder = bool(self.finetune_config.get("train_full_encoder", True))
        unfreeze_top_k_layers = int(self.finetune_config.get("unfreeze_top_k_layers", 0))

        if freeze_embeddings and hasattr(self.encoder, "embeddings"):
            self._freeze_module(self.encoder.embeddings)

        if train_full_encoder:
            return

        # Freeze everything first
        self._freeze_module(self.encoder)

        # Optionally keep embeddings frozen while opening top-k encoder blocks
        layers = _get_encoder_layers(self.encoder)
        if layers is not None and unfreeze_top_k_layers > 0:
            k = min(unfreeze_top_k_layers, len(layers))
            for layer in layers[-k:]:
                self._unfreeze_module(layer)

        # Pooler can stay trainable if present
        if hasattr(self.encoder, "pooler") and self.encoder.pooler is not None:
            self._unfreeze_module(self.encoder.pooler)

        # Respect explicit embedding freeze even if pooler/layers were reopened
        if freeze_embeddings and hasattr(self.encoder, "embeddings"):
            self._freeze_module(self.encoder.embeddings)

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if self.pooling == "mean":
            if attention_mask is None:
                return last_hidden_state.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).float()
            masked = last_hidden_state * mask
            return masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        # default: CLS token
        return last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> ClassifierOutput:
        encoder_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids
        encoder_kwargs.update(kwargs)

        if self.encoder_trainable:
            outputs = self.encoder(**encoder_kwargs)
        else:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(**encoder_kwargs)

        features = self._pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(features)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        return ClassifierOutput(
            loss=None,
            logits=logits,
            probs=probs,
            preds=preds,
            labels=labels,
            embeddings=features,
        )