from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.text.trainer import TextClassificationDataset


def test_text_dataset_reads_vnemos_compat_schema(tmp_path: Path) -> None:
    csv_path = tmp_path / "vnemos_train.csv"
    df = pd.DataFrame(
        [
            {
                "utterance_id": "utt_a",
                "sample_id": "utt_a",
                "id": "utt_a",
                "emotion": "anger",
                "orig_label": "angry",
                "label": "anger",
                "label_id": 0,
                "audio_path": "data/wavs16k/utt_a.wav",
                "path": "data/wavs16k/utt_a.wav",
                "filename": "utt_a.wav",
                "source_group": "scene_a",
                "group_id": "anger::scene_a",
                "duration": 3.2,
                "transcript_final": "toi rat buc minh",
                "text_for_model": "toi rat buc minh",
                "text": "toi rat buc minh",
                "normalized_text": "toi rat buc minh",
                "split": "train",
            }
        ]
    )
    df.to_csv(csv_path, index=False)

    dataset = TextClassificationDataset(
        csv_path=csv_path,
        dataset_config={
            "id_col": "sample_id",
            "text_col": "text",
            "raw_text_col": "transcript_final",
            "normalized_text_col": "normalized_text",
            "audio_path_col": "audio_path",
            "group_id_col": "group_id",
            "label_col": "label",
            "label_id_col": "label_id",
            "num_classes": 5,
            "use_normalized_text": False,
        },
        preprocessing_config={
            "normalize_whitespace": True,
            "lowercase": False,
            "word_segment": False,
        },
        project_root=".",
    )

    assert len(dataset) == 1
    item = dataset[0]
    assert item["id"] == "utt_a"
    assert item["raw_text"] == "toi rat buc minh"
    assert item["audio_path"] == "data/wavs16k/utt_a.wav"
    assert item["group_id"] == "anger::scene_a"
    assert item["label_id"] == 0