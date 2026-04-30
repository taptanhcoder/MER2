from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, file_path)


def load_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


class CheckpointManager:
    def __init__(
        self,
        run_dir: str | Path,
        monitor: str = "macro_f1",
        mode: str = "max",
        min_delta: float = 0.0,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.monitor = monitor
        self.mode = mode
        self.min_delta = float(min_delta)

        self.best_path = self.run_dir / "best.ckpt"
        self.last_path = self.run_dir / "last.ckpt"

        self.best_value: float | None = None
        self.best_epoch: int | None = None

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True

        if self.mode == "min":
            return value < (self.best_value - self.min_delta)

        return value > (self.best_value + self.min_delta)

    def save_last(self, state: dict[str, Any]) -> None:
        save_checkpoint(self.last_path, state)

    def save_best_if_improved(
        self,
        state: dict[str, Any],
        value: float,
        epoch: int | None = None,
    ) -> bool:
        if self._is_better(value):
            self.best_value = value
            self.best_epoch = epoch
            save_checkpoint(self.best_path, state)
            return True
        return False