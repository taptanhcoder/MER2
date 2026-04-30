from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.text.trainer import TextClassificationDataset


def test_text_dataset_can_fallback_from_label_to_label_id(tmp_path: Path) -> None:
    csv_path = tmp_path / "toy.csv"
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "text": "xin chao",
                "normalized_text": "xin chao",
                "label": "anger",
            },
            {
                "id": "b",
                "text": "rat vui",
                "normalized_text": "rat vui",
                "label": "enjoyment",
            },
        ]
    )
    df.to_csv(csv_path, index=False)

    dataset = TextClassificationDataset(
        csv_path=csv_path,
        dataset_config={
            "id_col": "id",
            "text_col": "text",
            "normalized_text_col": "normalized_text",
            "label_col": "label",
            "label_id_col": "label_id",
            "use_normalized_text": False,
        },
        preprocessing_config={
            "normalize_whitespace": True,
            "lowercase": False,
            "word_segment": False,
        },
        label_space_config={
            "labels": [
                "anger",
                "fear",
                "enjoyment",
                "sadness",
                "disgust",
                "surprise",
                "other",
            ],
            "label2id": {
                "anger": 0,
                "fear": 1,
                "enjoyment": 2,
                "sadness": 3,
                "disgust": 4,
                "surprise": 5,
                "other": 6,
            },
        },
    )

    assert len(dataset) == 2
    item0 = dataset[0]
    item1 = dataset[1]
    assert item0["label_id"] == 0
    assert item1["label_id"] == 2