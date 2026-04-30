from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.calibration.base import BaseCalibrator


class TemperatureScaler(BaseCalibrator):
    def __init__(
        self,
        init_temperature: float = 1.0,
        min_temperature: float = 0.05,
        max_temperature: float = 10.0,
        optimizer_config: dict[str, Any] | None = None,
    ) -> None:
        self.temperature = float(init_temperature)
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.optimizer_config = optimizer_config or {
            "name": "lbfgs",
            "lr": 0.01,
            "max_iter": 100,
            "tolerance_grad": 1e-7,
            "tolerance_change": 1e-9,
            "line_search_fn": "strong_wolfe",
        }

    def _clip_temperature(self, value: float) -> float:
        return float(max(self.min_temperature, min(self.max_temperature, value)))

    def fit(
        self,
        logits: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
    ) -> "TemperatureScaler":
        logits_tensor = (
            torch.from_numpy(np.asarray(logits).copy()).float()
            if isinstance(logits, np.ndarray)
            else logits.detach().float().cpu()
        )
        labels_tensor = (
            torch.from_numpy(np.asarray(labels).copy()).long()
            if isinstance(labels, np.ndarray)
            else labels.detach().long().cpu()
        )

        init_t = self._clip_temperature(self.temperature)
        log_temperature = torch.nn.Parameter(
            torch.log(torch.tensor(init_t, dtype=torch.float32))
        )

        optimizer_name = str(self.optimizer_config.get("name", "lbfgs")).lower()
        if optimizer_name != "lbfgs":
            raise ValueError(
                f"Unsupported calibration optimizer: {optimizer_name}. Only 'lbfgs' is supported."
            )

        optimizer = torch.optim.LBFGS(
            [log_temperature],
            lr=float(self.optimizer_config.get("lr", 0.01)),
            max_iter=int(self.optimizer_config.get("max_iter", 100)),
            tolerance_grad=float(self.optimizer_config.get("tolerance_grad", 1e-7)),
            tolerance_change=float(self.optimizer_config.get("tolerance_change", 1e-9)),
            line_search_fn=self.optimizer_config.get("line_search_fn", "strong_wolfe"),
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temperature = torch.clamp(
                torch.exp(log_temperature),
                min=self.min_temperature,
                max=self.max_temperature,
            )
            scaled_logits = logits_tensor / temperature
            loss = F.cross_entropy(scaled_logits, labels_tensor)
            loss.backward()
            return loss

        optimizer.step(closure)

        learned_temperature = torch.clamp(
            torch.exp(log_temperature.detach()),
            min=self.min_temperature,
            max=self.max_temperature,
        ).item()
        self.temperature = self._clip_temperature(float(learned_temperature))
        return self

    def transform_logits(
        self,
        logits: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        temperature = self._clip_temperature(self.temperature)

        if isinstance(logits, np.ndarray):
            return logits / temperature

        return logits / temperature

    def state_dict(self) -> dict[str, Any]:
        return {
            "method": "temperature_scaling",
            "temperature": float(self.temperature),
            "min_temperature": float(self.min_temperature),
            "max_temperature": float(self.max_temperature),
            "optimizer": dict(self.optimizer_config),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.temperature = self._clip_temperature(float(state_dict["temperature"]))
        self.min_temperature = float(state_dict.get("min_temperature", self.min_temperature))
        self.max_temperature = float(state_dict.get("max_temperature", self.max_temperature))
        if "optimizer" in state_dict:
            self.optimizer_config = dict(state_dict["optimizer"])