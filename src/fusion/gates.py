from __future__ import annotations

import torch
import torch.nn as nn


class ClassWiseAdaptiveGate(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        mode: str = "three_way_classwise",
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.mode = str(mode).lower()

        if self.mode not in {"fixed_avg", "two_way", "three_way_classwise"}:
            raise ValueError(f"Unsupported gate mode: {mode}")

        if self.mode == "fixed_avg":
            self.mlp = None
            return

        num_outputs = 2 * self.num_classes if self.mode == "two_way" else 3 * self.num_classes

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, gate_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(gate_input.shape[0])

        if self.mode == "fixed_avg":
            alpha_text = torch.full(
                (batch_size, self.num_classes),
                0.5,
                dtype=gate_input.dtype,
                device=gate_input.device,
            )
            alpha_speech = torch.full_like(alpha_text, 0.5)
            alpha_interaction = torch.zeros_like(alpha_text)
            return alpha_text, alpha_speech, alpha_interaction

        gate_logits = self.mlp(gate_input)

        if self.mode == "two_way":
            gate_logits = gate_logits.view(batch_size, self.num_classes, 2)
            gate_weights = torch.softmax(gate_logits, dim=-1)
            alpha_text = gate_weights[..., 0]
            alpha_speech = gate_weights[..., 1]
            alpha_interaction = torch.zeros_like(alpha_text)
            return alpha_text, alpha_speech, alpha_interaction

        gate_logits = gate_logits.view(batch_size, self.num_classes, 3)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        alpha_text = gate_weights[..., 0]
        alpha_speech = gate_weights[..., 1]
        alpha_interaction = gate_weights[..., 2]
        return alpha_text, alpha_speech, alpha_interaction