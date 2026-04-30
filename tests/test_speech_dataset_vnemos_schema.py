from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.speech.trainer import SpeechClassificationDataset


def test_speech_dataset_reads_vnemos_compat_schema(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "vnemos_train.csv"
    df = pd.DataFrame(
        [
            {
                "sample_id": "utt_a",
                "audio_path": "data/wavs16k/utt_a.wav",
                "group_id": "anger::scene_a",
                "duration": 3.2,
                "text": "toi rat buc minh",
                "transcript_final": "toi rat buc minh",
                "label": "anger",
                "label_id": 0,
            }
        ]
    )
    df.to_csv(csv_path, index=False)

    monkeypatch.setattr(
        "src.speech.trainer.load_audio",
        lambda path, target_sample_rate, mono: (torch.zeros(target_sample_rate), target_sample_rate),
    )

    dataset = SpeechClassificationDataset(
        csv_path=csv_path,
        dataset_config={
            "id_col": "sample_id",
            "audio_path_col": "audio_path",
            "duration_col": "duration",
            "group_id_col": "group_id",
            "text_col": "text",
            "raw_text_col": "transcript_final",
            "label_col": "label",
            "label_id_col": "label_id",
            "num_classes": 5,
        },
        audio_config={
            "sample_rate": 16000,
            "mono": True,
            "max_duration_sec": 2.0,
            "num_train_views": 1,
            "num_eval_views": 3,
        },
        augment_config=None,
        split="train",
        project_root=".",
    )

    assert len(dataset) == 1
    item = dataset[0]
    assert item["id"] == "utt_a"
    assert item["audio_path"] == "data/wavs16k/utt_a.wav"
    assert item["group_id"] == "anger::scene_a"
    assert item["duration"] == 3.2
    assert item["label_id"] == 0
    assert len(item["segments"]) == 1