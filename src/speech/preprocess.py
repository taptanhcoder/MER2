from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_waveform(
    waveform: torch.Tensor,
    enabled: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    if not enabled:
        return waveform
    peak = waveform.abs().max().clamp_min(eps)
    return waveform / peak


def trim_waveform_silence(
    waveform: torch.Tensor,
    sample_rate: int,
    enabled: bool = False,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    energy_threshold_ratio: float = 0.08,
    margin_ms: float = 80.0,
) -> torch.Tensor:
    """
    Light leading/trailing silence trim using frame RMS.
    Only trims the head/tail. It does not cut internal silence.
    """
    if not enabled:
        return waveform
    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform [T], got shape={tuple(waveform.shape)}")
    if waveform.numel() == 0:
        return waveform

    frame_len = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    margin = max(0, int(round(sample_rate * margin_ms / 1000.0)))

    if waveform.numel() < frame_len:
        return waveform

    frames = waveform.unfold(0, frame_len, hop_len)
    rms = frames.pow(2).mean(dim=-1).sqrt()

    max_rms = rms.max()
    if float(max_rms.item()) <= 0.0:
        return waveform

    threshold = max_rms * float(energy_threshold_ratio)
    active = rms >= threshold

    if not bool(active.any().item()):
        return waveform

    active_idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
    first_frame = int(active_idx[0].item())
    last_frame = int(active_idx[-1].item())

    start = max(0, first_frame * hop_len - margin)
    end = min(waveform.numel(), last_frame * hop_len + frame_len + margin)

    trimmed = waveform[start:end].contiguous().float()
    return trimmed if trimmed.numel() > 0 else waveform


def crop_or_pad_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    max_duration_sec: float,
    train: bool = True,
    random_crop: bool = True,
    center_crop_eval: bool = True,
    pad_to_max_duration: bool = False,
    min_duration_sec: float | None = None,
) -> torch.Tensor:
    """
    Expects waveform shape [T].

    Behavior:
    - first ensure waveform has at least `min_duration_sec`
    - then crop to `max_duration_sec` if too long
    - optionally pad to `max_duration_sec`
    """
    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform [T], got shape={tuple(waveform.shape)}")

    max_num_samples = int(round(sample_rate * max_duration_sec))
    if max_num_samples <= 0:
        raise ValueError("max_duration_sec must be positive")

    min_num_samples = 0
    if min_duration_sec is not None:
        min_num_samples = int(round(sample_rate * min_duration_sec))
        min_num_samples = max(1, min(min_num_samples, max_num_samples))

    num_samples = waveform.numel()

    if min_num_samples > 0 and num_samples < min_num_samples:
        pad_right = min_num_samples - num_samples
        waveform = F.pad(waveform, (0, pad_right))
        num_samples = waveform.numel()

    if num_samples > max_num_samples:
        if train and random_crop:
            max_start = num_samples - max_num_samples
            start = int(torch.randint(0, max_start + 1, (1,)).item())
        elif center_crop_eval:
            start = max((num_samples - max_num_samples) // 2, 0)
        else:
            start = 0
        waveform = waveform[start : start + max_num_samples]

    elif pad_to_max_duration and num_samples < max_num_samples:
        pad_right = max_num_samples - num_samples
        waveform = F.pad(waveform, (0, pad_right))

    return waveform.contiguous().float()


def build_random_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    max_duration_sec: float,
    num_crops: int = 2,
    min_duration_sec: float | None = None,
) -> list[torch.Tensor]:
    """
    Random training segments for long clips.
    If waveform is shorter than max_duration_sec, returns a single waveform.
    """
    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform [T], got shape={tuple(waveform.shape)}")

    max_num_samples = int(round(sample_rate * max_duration_sec))
    if waveform.numel() <= max_num_samples:
        return [
            crop_or_pad_waveform(
                waveform=waveform,
                sample_rate=sample_rate,
                max_duration_sec=max_duration_sec,
                train=True,
                random_crop=False,
                center_crop_eval=False,
                pad_to_max_duration=False,
                min_duration_sec=min_duration_sec,
            )
        ]

    num_crops = max(1, int(num_crops))
    segments: list[torch.Tensor] = []
    max_start = waveform.numel() - max_num_samples

    for _ in range(num_crops):
        start = int(torch.randint(0, max_start + 1, (1,)).item())
        segment = waveform[start : start + max_num_samples].contiguous().float()
        segments.append(segment)

    return segments


def build_dynamic_inference_segments(
    waveform: torch.Tensor,
    sample_rate: int,
    max_duration_sec: float,
    medium_num_crops: int = 3,
    long_num_crops: int = 5,
    long_factor_threshold: float = 2.0,
    min_duration_sec: float | None = None,
) -> list[torch.Tensor]:
    """
    Deterministic multi-crop inference segments.

    Rules:
    - duration <= max_duration_sec: 1 crop
    - max_duration_sec < duration <= long_factor_threshold * max_duration_sec: medium_num_crops
    - duration > long_factor_threshold * max_duration_sec: long_num_crops
    """
    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform [T], got shape={tuple(waveform.shape)}")

    max_num_samples = int(round(sample_rate * max_duration_sec))
    if max_num_samples <= 0:
        raise ValueError("max_duration_sec must be positive")

    waveform = crop_or_pad_waveform(
        waveform=waveform,
        sample_rate=sample_rate,
        max_duration_sec=max_duration_sec,
        train=False,
        random_crop=False,
        center_crop_eval=True,
        pad_to_max_duration=False,
        min_duration_sec=min_duration_sec,
    )

    if waveform.numel() <= max_num_samples:
        return [waveform]

    # Recompute on the original long waveform path
    num_samples = waveform.numel()
    # If crop_or_pad_waveform already shortened it, caller should pass original waveform.
    # In practice we handle this in the caller by only using this function for long clips.
    if num_samples <= max_num_samples:
        return [waveform]

    raise RuntimeError(
        "build_dynamic_inference_segments should be called with the original long waveform. "
        "Use build_dynamic_inference_segments_from_long_waveform instead."
    )


def build_dynamic_inference_segments_from_long_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    max_duration_sec: float,
    medium_num_crops: int = 3,
    long_num_crops: int = 5,
    long_factor_threshold: float = 2.0,
    min_duration_sec: float | None = None,
) -> list[torch.Tensor]:
    """
    Deterministic multi-crop inference for long waveforms.
    """
    if waveform.dim() != 1:
        raise ValueError(f"Expected 1D waveform [T], got shape={tuple(waveform.shape)}")

    max_num_samples = int(round(sample_rate * max_duration_sec))
    if max_num_samples <= 0:
        raise ValueError("max_duration_sec must be positive")

    waveform = waveform.contiguous().float()

    min_num_samples = 0
    if min_duration_sec is not None:
        min_num_samples = int(round(sample_rate * min_duration_sec))
        min_num_samples = max(1, min(min_num_samples, max_num_samples))

    if min_num_samples > 0 and waveform.numel() < min_num_samples:
        waveform = F.pad(waveform, (0, min_num_samples - waveform.numel()))

    num_samples = waveform.numel()

    if num_samples <= max_num_samples:
        return [
            crop_or_pad_waveform(
                waveform=waveform,
                sample_rate=sample_rate,
                max_duration_sec=max_duration_sec,
                train=False,
                random_crop=False,
                center_crop_eval=True,
                pad_to_max_duration=False,
                min_duration_sec=min_duration_sec,
            )
        ]

    if num_samples <= int(round(long_factor_threshold * max_num_samples)):
        num_crops = max(1, int(medium_num_crops))
    else:
        num_crops = max(1, int(long_num_crops))

    max_start = num_samples - max_num_samples

    if num_crops == 1 or max_start == 0:
        start_positions = [max_start // 2]
    else:
        start_positions = torch.linspace(
            0,
            max_start,
            steps=num_crops,
            dtype=torch.float32,
        ).round().to(torch.long).tolist()

    deduped_positions: list[int] = []
    seen = set()
    for pos in start_positions:
        pos_int = int(pos)
        if pos_int not in seen:
            deduped_positions.append(pos_int)
            seen.add(pos_int)

    segments = [
        waveform[start : start + max_num_samples].contiguous().float()
        for start in deduped_positions
    ]
    return segments