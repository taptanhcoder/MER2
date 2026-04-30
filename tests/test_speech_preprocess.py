from __future__ import annotations

import torch

from src.speech.preprocess import (
    build_dynamic_inference_segments_from_long_waveform,
    build_random_segments,
    crop_or_pad_waveform,
    normalize_waveform,
    trim_waveform_silence,
)


def test_crop_or_pad_waveform_respects_max_duration() -> None:
    waveform = torch.arange(0, 32000, dtype=torch.float32)
    out = crop_or_pad_waveform(
        waveform=waveform,
        sample_rate=16000,
        max_duration_sec=1.0,
        train=False,
        random_crop=False,
        center_crop_eval=True,
        pad_to_max_duration=False,
        min_duration_sec=0.25,
    )
    assert out.numel() == 16000


def test_trim_waveform_silence_trims_head_and_tail() -> None:
    silence = torch.zeros(1600, dtype=torch.float32)
    speech = torch.ones(3200, dtype=torch.float32) * 0.5
    waveform = torch.cat([silence, speech, silence], dim=0)

    trimmed = trim_waveform_silence(
        waveform=waveform,
        sample_rate=16000,
        enabled=True,
        frame_ms=25,
        hop_ms=10,
        energy_threshold_ratio=0.1,
        margin_ms=0,
    )

    assert trimmed.numel() < waveform.numel()
    assert trimmed.numel() >= speech.numel()


def test_build_random_segments_for_long_clip_returns_two_segments() -> None:
    waveform = torch.arange(0, 16000 * 12, dtype=torch.float32)
    segments = build_random_segments(
        waveform=waveform,
        sample_rate=16000,
        max_duration_sec=8.0,
        num_crops=2,
        min_duration_sec=0.25,
    )
    assert len(segments) == 2
    assert all(seg.numel() == 16000 * 8 for seg in segments)


def test_build_dynamic_inference_segments_returns_five_for_very_long_audio() -> None:
    waveform = torch.arange(0, 16000 * 25, dtype=torch.float32)
    segments = build_dynamic_inference_segments_from_long_waveform(
        waveform=waveform,
        sample_rate=16000,
        max_duration_sec=8.0,
        medium_num_crops=3,
        long_num_crops=5,
        long_factor_threshold=2.0,
        min_duration_sec=0.25,
    )
    assert len(segments) == 5
    assert all(seg.numel() == 16000 * 8 for seg in segments)


def test_normalize_waveform_peak_normalization() -> None:
    waveform = torch.tensor([0.0, 2.0, -1.0], dtype=torch.float32)
    out = normalize_waveform(waveform, enabled=True)
    assert float(out.abs().max()) == 1.0