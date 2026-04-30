import wave
from pathlib import Path

import numpy as np
import pytest

from src.speech.audio_io import load_audio


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    int16 = (samples * 32767.0).astype(np.int16)

    if int16.ndim == 1:
        channels = 1
    else:
        channels = int16.shape[1]

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16.tobytes())


def test_load_audio_converts_stereo_to_mono(tmp_path: Path):
    sr = 16000
    time = np.linspace(0.0, 0.1, int(sr * 0.1), endpoint=False)
    left = 0.1 * np.sin(2 * np.pi * 440 * time)
    right = 0.1 * np.sin(2 * np.pi * 660 * time)
    stereo = np.stack([left, right], axis=1)

    wav_path = tmp_path / "stereo.wav"
    _write_wav(wav_path, stereo, sr)

    waveform, loaded_sr = load_audio(wav_path, target_sample_rate=sr, mono=True)
    assert loaded_sr == sr
    assert waveform.ndim == 1
    assert waveform.numel() == stereo.shape[0]


def test_load_audio_resamples(tmp_path: Path):
    original_sr = 8000
    target_sr = 16000
    time = np.linspace(0.0, 0.1, int(original_sr * 0.1), endpoint=False)
    mono = 0.1 * np.sin(2 * np.pi * 220 * time)

    wav_path = tmp_path / "mono.wav"
    _write_wav(wav_path, mono, original_sr)

    waveform, loaded_sr = load_audio(wav_path, target_sample_rate=target_sr, mono=True)
    assert loaded_sr == target_sr
    assert waveform.ndim == 1
    assert waveform.numel() > mono.shape[0]


def test_load_audio_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "missing.wav"
    with pytest.raises(FileNotFoundError):
        load_audio(missing)