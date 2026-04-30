from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.speech.trainer import SpeechClassificationDataset


def test_speech_dataset_fallback_to_zero_waveform_on_load_error(
    tmp_path: Path,
    monkeypatch,
) -> None:
    csv_path = tmp_path / "toy.csv"
    df = pd.DataFrame(
        [
            {
                "path": "/tmp/not_real.wav",
                "filename": "not_real.wav",
                "group_id": "anger::scene_a",
                "label": "anger",
                "label_id": 0,
            }
        ]
    )
    df.to_csv(csv_path, index=False)

    def _raise_load_error(*args, **kwargs):
        raise RuntimeError("audio decode failed")

    monkeypatch.setattr("src.speech.trainer.load_audio", _raise_load_error)

    dataset = SpeechClassificationDataset(
        csv_path=csv_path,
        dataset_config={
            "path_col": "path",
            "filename_col": "filename",
            "group_id_col": "group_id",
            "label_col": "label",
            "label_id_col": "label_id",
            "num_classes": 5,
        },
        audio_config={
            "sample_rate": 16000,
            "mono": True,
            "max_duration_sec": 8.0,
            "min_duration_sec": 0.25,
            "normalize_waveform": False,
            "fail_on_load_error": False,
            "trim_silence": {"enabled": False},
        },
        augment_config=None,
        split="train",
    )

    item = dataset[0]
    assert isinstance(item["waveform"], torch.Tensor)
    assert item["waveform"].numel() == int(16000 * 0.25)
    assert float(item["waveform"].abs().sum()) == 0.0