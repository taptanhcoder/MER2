import wave
from pathlib import Path

import numpy as np
import pandas as pd

from src.speech.trainer import SpeechClassificationDataset


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    int16 = (samples * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(int16.tobytes())


def test_speech_dataset_reads_manifest_and_sample(tmp_path: Path):
    sr = 16000
    wav_path = tmp_path / "sample.wav"
    signal = 0.1 * np.sin(2 * np.pi * 440 * np.linspace(0.0, 0.1, int(sr * 0.1), endpoint=False))
    _write_wav(wav_path, signal, sr)

    manifest = pd.DataFrame(
        [
            {
                "id": "vn_0001",
                "path": str(wav_path),
                "filename": "sample.wav",
                "group_id": "anger::spk1",
                "label": "anger",
                "label_id": 0,
            }
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    dataset_config = {
        "columns": {
            "path_col": "path",
            "filename_col": "filename",
            "group_id_col": "group_id",
            "label_col": "label",
            "label_id_col": "label_id",
        },
        "audio": {
            "target_sample_rate": 16000,
            "mono": True,
            "normalize": False,
            "max_duration_seconds": None,
        },
    }

    dataset = SpeechClassificationDataset(
        manifest_path=manifest_path,
        dataset_config=dataset_config,
        split="train",
    )

    sample = dataset[0]
    assert sample["sample_id"] == "vn_0001"
    assert sample["label_id"] == 0
    assert sample["group_id"] == "anger::spk1"
    assert sample["waveform"].ndim == 1
    assert sample["sampling_rate"] == 16000
    assert sample["raw_length"] > 0
    assert dataset.class_counts == [1]