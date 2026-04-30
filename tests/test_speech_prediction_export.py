from __future__ import annotations

from src.common.interfaces import EpochResult, PredictionBundle
from src.speech.trainer import epoch_result_to_dataframe


def test_speech_prediction_export_contains_metadata() -> None:
    result = EpochResult(
        split="test",
        loss=1.23,
        metrics={"macro_f1": 0.5},
        predictions=PredictionBundle(
            ids=["utt_a"],
            labels=[0],
            preds=[1],
            probs=[[0.2, 0.8]],
            logits=[[0.1, 1.2]],
            metadata={
                "audio_paths": ["data/wavs16k/utt_a.wav"],
                "group_ids": ["anger::scene_a"],
                "durations": [3.2],
                "texts": ["toi rat buc minh"],
                "raw_texts": ["toi rat buc minh"],
            },
        ),
    )

    df = epoch_result_to_dataframe(result, label_names=["anger", "fear"])

    assert df.shape[0] == 1
    assert df.loc[0, "sample_id"] == "utt_a"
    assert df.loc[0, "audio_path"] == "data/wavs16k/utt_a.wav"
    assert df.loc[0, "group_id"] == "anger::scene_a"
    assert float(df.loc[0, "duration"]) == 3.2
    assert df.loc[0, "text"] == "toi rat buc minh"
    assert df.loc[0, "raw_text"] == "toi rat buc minh"
    assert df.loc[0, "label"] == "anger"
    assert df.loc[0, "pred"] == "fear"