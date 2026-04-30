from __future__ import annotations

from pathlib import Path

from src.data.manifest_reader import (
    canonicalize_vnemos_alignment_key,
    prepare_vnemos_paired_from_config,
)
from src.utils.io import write_jsonl


def test_canonicalize_vnemos_alignment_key_strips_copy_and_normalizes() -> None:
    value = "Copy of Angry_SCVMC12-00.16.21.029-00.16.25.280-seg5.wav"
    key = canonicalize_vnemos_alignment_key(value)
    assert key == "angry_scvmc12_00_16_21_029_00_16_25_280_seg5"


def test_prepare_vnemos_paired_can_align_by_filename_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_root = tmp_path / "data" / "VNEMOS" / "angry"
    raw_root.mkdir(parents=True, exist_ok=True)
    raw_audio = raw_root / "Copy of Angry_SCVMC12-00.16.21.029-00.16.25.280-seg5.wav"
    raw_audio.write_bytes(b"fake")

    transcripts_dir = tmp_path / "data" / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    master_jsonl = transcripts_dir / "vnemos_mer_all.jsonl"
    write_jsonl(
        [
            {
                "filename": "Angry_SCVMC12-00.16.21.029-00.16.25.280-seg5.wav",
                "transcript_final": "toi rat tuc gian",
            }
        ],
        master_jsonl,
    )

    processed_root = tmp_path / "data" / "wavs16k"

    def _fake_prepare_processed_audio(
        raw_audio_path: Path,
        processed_audio_path: Path,
        target_sample_rate: int,
        mono: bool,
        overwrite: bool,
    ) -> float:
        processed_audio_path.parent.mkdir(parents=True, exist_ok=True)
        processed_audio_path.write_bytes(b"fake-processed")
        return 1.23

    monkeypatch.setattr(
        "src.data.manifest_reader._prepare_processed_audio",
        _fake_prepare_processed_audio,
    )

    config = {
        "split": {
            "seed": 42,
            "train_ratio": 0.8,
            "valid_ratio": 0.1,
            "test_ratio": 0.1,
        },
        "dataset": {
            "name": "vnemos_paired_mer",
            "raw_audio_root": str(tmp_path / "data" / "VNEMOS"),
            "processed_audio_root": str(processed_root),
            "transcripts_dir": str(transcripts_dir),
            "master_transcript_file": str(master_jsonl),
            "allowed_extensions": [".wav"],
            "transcript_field": "transcript_final",
            "text_field": "text_for_model",
            "transcript_alignment": {
                "preferred_field": None,
                "candidates": ["utterance_id", "filename", "id"],
            },
            "fail_on_missing_transcript": True,
            "fail_on_alignment_collision": True,
            "write_compatibility_csv": True,
            "compatibility_csv_dir": str(tmp_path / "data" / "splits"),
        },
        "audio_preprocess": {
            "sample_rate": 16000,
            "mono": True,
            "output_format": "pcm_16",
        },
    }

    prepared = prepare_vnemos_paired_from_config(config, overwrite_audio=True)

    assert prepared.validation_errors == []
    assert prepared.alignment_report["selected_transcript_alignment_field"] == "filename"
    assert len(prepared.all_frame) == 1
    assert prepared.all_frame.iloc[0]["utterance_id"] == "angry_scvmc12_00_16_21_029_00_16_25_280_seg5"
    assert prepared.all_frame.iloc[0]["transcript_final"] == "toi rat tuc gian"