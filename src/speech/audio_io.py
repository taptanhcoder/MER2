from __future__ import annotations

from pathlib import Path

import torch

try:
    import torchaudio
except ImportError:  # pragma: no cover
    torchaudio = None

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

try:
    import librosa
except ImportError:  # pragma: no cover
    librosa = None


def _load_with_torchaudio(path: str) -> tuple[torch.Tensor, int]:
    if torchaudio is None:
        raise ImportError("torchaudio is not available")
    waveform, sample_rate = torchaudio.load(path)
    return waveform.float(), int(sample_rate)


def _load_with_soundfile(path: str) -> tuple[torch.Tensor, int]:
    if sf is None:
        raise ImportError("soundfile is not available")
    array, sample_rate = sf.read(path, always_2d=True)
    waveform = torch.from_numpy(array.T).float()
    return waveform, int(sample_rate)


def _resample(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform

    if torchaudio is not None:
        return torchaudio.functional.resample(
            waveform,
            orig_freq=orig_sr,
            new_freq=target_sr,
        )

    if librosa is None:
        raise ImportError("Need torchaudio or librosa for resampling.")

    channels: list[torch.Tensor] = []
    for channel_idx in range(waveform.size(0)):
        channel = waveform[channel_idx].cpu().numpy()
        resampled = librosa.resample(channel, orig_sr=orig_sr, target_sr=target_sr)
        channels.append(torch.from_numpy(resampled).float())
    return torch.stack(channels, dim=0)


def load_audio(
    path: str,
    target_sample_rate: int,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        waveform, sample_rate = _load_with_torchaudio(str(audio_path))
    except Exception:
        waveform, sample_rate = _load_with_soundfile(str(audio_path))

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    if mono and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    waveform = _resample(
        waveform=waveform,
        orig_sr=sample_rate,
        target_sr=target_sample_rate,
    )

    if mono and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform.contiguous().float(), int(target_sample_rate)


def get_audio_duration_sec(path: str) -> float:
    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if torchaudio is not None:
        try:
            info = torchaudio.info(str(audio_path))
            if info.sample_rate > 0:
                return float(info.num_frames) / float(info.sample_rate)
        except Exception:
            pass

    if sf is not None:
        try:
            info = sf.info(str(audio_path))
            if info.samplerate > 0:
                return float(info.frames) / float(info.samplerate)
        except Exception:
            pass

    if librosa is not None:
        try:
            return float(librosa.get_duration(path=str(audio_path)))
        except Exception:
            pass

    waveform, sample_rate = load_audio(str(audio_path), target_sample_rate=16000, mono=True)
    return float(waveform.shape[-1]) / 16000.0