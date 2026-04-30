from __future__ import annotations

import random
from typing import Any, Callable

import torch


class AddGaussianNoise:
    def __init__(self, std: float) -> None:
        self.std = float(std)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return waveform
        noise = torch.randn_like(waveform) * self.std
        return waveform + noise


class RandomGain:
    def __init__(self, min_gain: float, max_gain: float) -> None:
        self.min_gain = float(min_gain)
        self.max_gain = float(max_gain)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        gain = random.uniform(self.min_gain, self.max_gain)
        return waveform * gain


class ComposeWaveformAugment:
    def __init__(self, transforms: list[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self.transforms = transforms

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        output = waveform
        for transform in self.transforms:
            output = transform(output)
        return output


def build_waveform_augment(config: dict[str, Any] | None):
    if not config:
        return None

    if not bool(config.get("enabled", True)):
        return None

    transforms: list[Callable[[torch.Tensor], torch.Tensor]] = []

    noise_std = float(config.get("gaussian_noise_std", 0.0))
    if noise_std > 0:
        transforms.append(AddGaussianNoise(std=noise_std))

    min_gain = float(config.get("min_gain", 1.0))
    max_gain = float(config.get("max_gain", 1.0))
    if min_gain != 1.0 or max_gain != 1.0:
        transforms.append(RandomGain(min_gain=min_gain, max_gain=max_gain))

    if not transforms:
        return None

    return ComposeWaveformAugment(transforms)