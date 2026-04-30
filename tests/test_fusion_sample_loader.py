from __future__ import annotations

import json
from pathlib import Path

from src.fusion.sample_loader import load_fusion_sample


def test_load_fusion_sample_resolves_audio_path(tmp_path: Path) -> None:
    audio_root = tmp_path / "VNEMOS"
    audio_dir = audio_root / "angry"
    audio_dir.mkdir(parents=True, exist_ok=True)

    wav_name = "Copy of Angry_scvmc15-00.33.11.861-00.33.18.082-seg8.wav"
    wav_path = audio_dir / wav_name
    wav_path.write_bytes(b"fake")

    sample_json = tmp_path / "test.json"
    sample_payload = {
        "utterance_id": "Copy_of_Angry_scvmc15-00.33.11.861-00.33.18.082-seg8",
        "speaker_id": wav_name,
        "start": 0.0,
        "end": 8.823625,
        "transcript": "Mẹ cứ bắt con phải đi khám thế nhở con phải nói bao lần nữa thì mẹ mới hiểu.",
        "emotion": "angry",
    }
    sample_json.write_text(json.dumps(sample_payload, ensure_ascii=False), encoding="utf-8")

    sample = load_fusion_sample(sample_json, audio_root=audio_root)

    assert sample.sample_id == sample_payload["utterance_id"]
    assert sample.raw_emotion == "angry"
    assert sample.label == "anger"
    assert sample.speaker_id == wav_name
    assert sample.audio_path == wav_path.resolve()
