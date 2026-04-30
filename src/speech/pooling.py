from __future__ import annotations

import torch
import torch.nn as nn


def _sanitize_attention_mask(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    if attention_mask is None:
        return torch.ones(batch_size, seq_len, dtype=torch.long, device=hidden_states.device)

    mask = attention_mask.long().to(hidden_states.device)
    invalid_rows = mask.sum(dim=1) <= 0
    if invalid_rows.any():
        mask = mask.clone()
        bad_indices = invalid_rows.nonzero(as_tuple=False).view(-1)
        for row_idx in bad_indices.tolist():
            mask[row_idx, 0] = 1
    return mask


class MeanPooling(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = _sanitize_attention_mask(hidden_states, attention_mask)
        mask_f = mask.unsqueeze(-1).float()
        summed = (hidden_states * mask_f).sum(dim=1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return summed / denom


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.scorer = nn.Linear(input_dim, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = _sanitize_attention_mask(hidden_states, attention_mask)
        scores = self.scorer(self.dropout(hidden_states)).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e4)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return pooled


def build_pooling(name: str, input_dim: int, dropout: float = 0.0) -> nn.Module:
    name = str(name).lower()
    if name == "mean":
        return MeanPooling()
    if name == "attn":
        return AttentionPooling(input_dim=input_dim, dropout=dropout)
    raise KeyError(f"Unsupported speech pooling: {name}")