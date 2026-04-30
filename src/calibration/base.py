from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class BaseCalibrator(ABC):
   @abstractmethod
   def fit(
       self,
       logits: np.ndarray | torch.Tensor,
       labels: np.ndarray | torch.Tensor,
   ) -> "BaseCalibrator":
       raise NotImplementedError

   @abstractmethod
   def transform_logits(
       self,
       logits: np.ndarray | torch.Tensor,
   ) -> np.ndarray | torch.Tensor:
       raise NotImplementedError

   def transform_probs(
       self,
       logits: np.ndarray | torch.Tensor,
   ) -> np.ndarray | torch.Tensor:
       scaled_logits = self.transform_logits(logits)
       if isinstance(scaled_logits, np.ndarray):
           shifted = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
           exp_values = np.exp(shifted)
           return exp_values / exp_values.sum(axis=1, keepdims=True)

       return torch.softmax(scaled_logits, dim=-1)

   @abstractmethod
   def state_dict(self) -> dict[str, Any]:
       raise NotImplementedError

   @abstractmethod
   def load_state_dict(self, state_dict: dict[str, Any]) -> None:
       raise NotImplementedError
