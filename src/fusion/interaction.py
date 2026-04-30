from __future__ import annotations

import math

import torch
import torch.nn as nn


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return sequence.mean(dim=1)

    mask = mask.unsqueeze(-1).float()
    summed = (sequence * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _ensure_mask(sequence: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return torch.ones(
            sequence.shape[0],
            sequence.shape[1],
            dtype=torch.long,
            device=sequence.device,
        )
    return mask.long()


def _sanitize_bank(
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.nan_to_num(tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mask = (mask > 0).long()

    invalid_rows = mask.sum(dim=1) <= 0
    if invalid_rows.any():
        tokens = tokens.clone()
        mask = mask.clone()
        bad_indices = invalid_rows.nonzero(as_tuple=False).view(-1)
        for row_idx in bad_indices.tolist():
            mask[row_idx, 0] = 1
            tokens[row_idx, 0].zero_()

    tokens = tokens * mask.unsqueeze(-1).float()
    return tokens.contiguous(), mask.contiguous()


class _CompactCrossAttentionDirection(nn.Module):
    """
    Stable cross-attention over compact token banks.

    Important design choice:
    - We do NOT use batched GEMM (`torch.bmm`) over the full batch here.
    - We run attention sample-by-sample with 2D matmul.
    This is intentional: compact token banks are tiny, and correctness/stability
    are more important than batched GEMM throughput in this research-stage fusion block.
    """

    def __init__(
        self,
        fusion_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.fusion_dim = int(fusion_dim)
        self.scale = math.sqrt(float(self.fusion_dim))

        self.q_proj = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.k_proj = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.v_proj = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.out_proj = nn.Linear(self.fusion_dim, self.fusion_dim)
        self.dropout = nn.Dropout(float(dropout))

    def _single_sample_attention(
        self,
        q: torch.Tensor,       # [Kq, D]
        q_mask: torch.Tensor,  # [Kq]
        k: torch.Tensor,       # [Kk, D]
        k_mask: torch.Tensor,  # [Kk]
        v: torch.Tensor,       # [Kk, D]
    ) -> torch.Tensor:
        query_idx = (q_mask > 0).nonzero(as_tuple=False).view(-1)
        key_idx = (k_mask > 0).nonzero(as_tuple=False).view(-1)

        out = torch.zeros_like(q)

        if query_idx.numel() == 0 or key_idx.numel() == 0:
            return out

        q_valid = q.index_select(0, query_idx).contiguous()   # [Lq, D]
        k_valid = k.index_select(0, key_idx).contiguous()     # [Lk, D]
        v_valid = v.index_select(0, key_idx).contiguous()     # [Lk, D]

        scores = torch.matmul(q_valid, k_valid.transpose(0, 1).contiguous()) / self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx_valid = torch.matmul(attn, v_valid)

        out.index_copy_(0, query_idx, ctx_valid)
        return out

    def forward(
        self,
        query_tokens: torch.Tensor,
        query_mask: torch.Tensor | None,
        key_value_tokens: torch.Tensor,
        key_value_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        query_mask = _ensure_mask(query_tokens, query_mask)
        key_value_mask = _ensure_mask(key_value_tokens, key_value_mask)

        query_tokens, query_mask = _sanitize_bank(query_tokens, query_mask)
        key_value_tokens, key_value_mask = _sanitize_bank(key_value_tokens, key_value_mask)

        q = self.q_proj(query_tokens)
        k = self.k_proj(key_value_tokens)
        v = self.v_proj(key_value_tokens)

        context_rows: list[torch.Tensor] = []
        batch_size = int(q.shape[0])

        for i in range(batch_size):
            context_i = self._single_sample_attention(
                q=q[i],
                q_mask=query_mask[i],
                k=k[i],
                k_mask=key_value_mask[i],
                v=v[i],
            )
            context_rows.append(context_i)

        context = torch.stack(context_rows, dim=0)
        context = self.out_proj(context)

        query_mask_f = query_mask.unsqueeze(-1).float()
        context = context * query_mask_f
        return context


class BiDirectionalCrossAttention(nn.Module):
    """
    Proposal-aligned compact sequence interaction:
    - text compact token bank attends to speech compact token bank
    - speech compact token bank attends to text compact token bank
    - pooled interaction representation -> interaction evidence
    """

    def __init__(
        self,
        text_dim: int,
        speech_dim: int,
        fusion_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.speech_proj = nn.Linear(speech_dim, fusion_dim)

        self.text_to_speech = _CompactCrossAttentionDirection(
            fusion_dim=fusion_dim,
            dropout=dropout,
        )
        self.speech_to_text = _CompactCrossAttentionDirection(
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

        self.text_norm = nn.LayerNorm(fusion_dim)
        self.speech_norm = nn.LayerNorm(fusion_dim)
        self.combine = nn.Linear(fusion_dim * 2, fusion_dim)

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor | None,
        speech_tokens: torch.Tensor,
        speech_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_tokens = torch.nan_to_num(text_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
        speech_tokens = torch.nan_to_num(speech_tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)

        text_seq = self.text_proj(text_tokens)
        speech_seq = self.speech_proj(speech_tokens)

        text_context = self.text_to_speech(
            query_tokens=text_seq,
            query_mask=text_mask,
            key_value_tokens=speech_seq,
            key_value_mask=speech_mask,
        )
        speech_context = self.speech_to_text(
            query_tokens=speech_seq,
            query_mask=speech_mask,
            key_value_tokens=text_seq,
            key_value_mask=text_mask,
        )

        text_mask = _ensure_mask(text_context, text_mask)
        speech_mask = _ensure_mask(speech_context, speech_mask)

        text_context = self.text_norm(text_seq + text_context)
        speech_context = self.speech_norm(speech_seq + speech_context)

        pooled_text = masked_mean(text_context, text_mask)
        pooled_speech = masked_mean(speech_context, speech_mask)

        interaction_embedding = self.combine(torch.cat([pooled_text, pooled_speech], dim=-1))
        return interaction_embedding, torch.cat([pooled_text, pooled_speech], dim=-1)