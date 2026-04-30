from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModel

from src.common.interfaces import ClassifierOutput
from src.common.registry import MODEL_REGISTRY
from src.speech.heads import ClassificationHead
from src.speech.pooling import build_pooling


def _extract_model_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if "model" in config:
        return dict(config["model"]), dict(config.get("finetune", {}))
    return dict(config), {}


def _get_hubert_layers(encoder: torch.nn.Module):
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layers"):
        return encoder.encoder.layers
    if hasattr(encoder, "layers"):
        return encoder.layers
    return None


@MODEL_REGISTRY.register("hubert-base")
@MODEL_REGISTRY.register("hubert-base-ls960")
class HuBERTClassifier(torch.nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.model_config, self.finetune_config = _extract_model_config(config)

        pretrained_name = self.model_config["pretrained_name"]
        self.pooling_name = str(self.model_config.get("pooling", "attn")).lower()
        self.num_classes = int(self.model_config["num_classes"])
        self.dropout_p = float(self.model_config.get("dropout", 0.1))

        hf_config = AutoConfig.from_pretrained(pretrained_name)
        self.encoder = AutoModel.from_pretrained(pretrained_name, config=hf_config)

        if bool(self.model_config.get("gradient_checkpointing", False)) and hasattr(
            self.encoder, "gradient_checkpointing_enable"
        ):
            self.encoder.gradient_checkpointing_enable()

        hidden_size = int(getattr(hf_config, "hidden_size"))

        self.pooling = build_pooling(
            name=self.pooling_name,
            input_dim=hidden_size,
            dropout=float(self.model_config.get("pooling_dropout", self.dropout_p)),
        )
        self.dropout = torch.nn.Dropout(float(self.model_config.get("classifier_dropout", self.dropout_p)))
        self.classifier = ClassificationHead(
            input_dim=hidden_size,
            num_classes=self.num_classes,
            dropout=float(self.model_config.get("classifier_dropout", self.dropout_p)),
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
        freeze_feature_extractor = bool(self.finetune_config.get("freeze_feature_extractor", True))
        train_full_encoder = bool(self.finetune_config.get("train_full_encoder", True))
        unfreeze_top_k_layers = int(self.finetune_config.get("unfreeze_top_k_layers", 0))
        unfreeze_feature_projection = bool(self.finetune_config.get("unfreeze_feature_projection", True))
        unfreeze_positional_conv = bool(self.finetune_config.get("unfreeze_positional_conv", True))

        if freeze_feature_extractor and hasattr(self.encoder, "feature_extractor"):
            if hasattr(self.encoder.feature_extractor, "_freeze_parameters"):
                self.encoder.feature_extractor._freeze_parameters()
            self._freeze_module(self.encoder.feature_extractor)

        if train_full_encoder:
            return

        self._freeze_module(self.encoder)

        if hasattr(self.encoder, "feature_projection") and unfreeze_feature_projection:
            self._unfreeze_module(self.encoder.feature_projection)

        if hasattr(self.encoder, "encoder"):
            if hasattr(self.encoder.encoder, "layer_norm"):
                self._unfreeze_module(self.encoder.encoder.layer_norm)
            if unfreeze_positional_conv and hasattr(self.encoder.encoder, "pos_conv_embed"):
                self._unfreeze_module(self.encoder.encoder.pos_conv_embed)

        layers = _get_hubert_layers(self.encoder)
        if layers is not None and unfreeze_top_k_layers > 0:
            k = min(unfreeze_top_k_layers, len(layers))
            for layer in layers[-k:]:
                self._unfreeze_module(layer)

        if freeze_feature_extractor and hasattr(self.encoder, "feature_extractor"):
            if hasattr(self.encoder.feature_extractor, "_freeze_parameters"):
                self.encoder.feature_extractor._freeze_parameters()
            self._freeze_module(self.encoder.feature_extractor)

    def _resolve_feature_attention_mask(
        self,
        input_attention_mask: torch.Tensor | None,
        feature_length: int,
    ) -> torch.Tensor | None:
        if input_attention_mask is None:
            return None

        if hasattr(self.encoder, "_get_feature_vector_attention_mask"):
            return self.encoder._get_feature_vector_attention_mask(
                feature_length,
                input_attention_mask,
            )
        return None

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> ClassifierOutput:
        encoder_kwargs: dict[str, Any] = {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        encoder_kwargs.update(kwargs)

        if self.encoder_trainable:
            outputs = self.encoder(**encoder_kwargs)
        else:
            self.encoder.eval()
            with torch.no_grad():
                outputs = self.encoder(**encoder_kwargs)

        hidden_states = outputs.last_hidden_state
        feature_mask = self._resolve_feature_attention_mask(
            input_attention_mask=attention_mask,
            feature_length=int(hidden_states.shape[1]),
        )

        pooled = self.pooling(hidden_states, attention_mask=feature_mask)
        logits = self.classifier(self.dropout(pooled))
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        return ClassifierOutput(
            loss=None,
            logits=logits,
            probs=probs,
            preds=preds,
            labels=labels,
            embeddings=pooled,
        )