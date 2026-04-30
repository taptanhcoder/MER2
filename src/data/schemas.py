from __future__ import annotations

TEXT_REQUIRED_COLUMNS = [
    "id",
    "text",
    "normalized_text",
    "orig_label",
    "label",
    "label_id",
]

SPEECH_REQUIRED_COLUMNS = [
    "path",
    "filename",
    "group_id",
    "label",
    "label_id",
]

PAIRED_JSONL_WRITE_FIELDS = [
    "utterance_id",
    "emotion",
    "audio_path",
    "duration",
    "transcript_final",
    "text_for_model",
    "split",
]

PAIRED_JSONL_REQUIRED_FIELDS = list(PAIRED_JSONL_WRITE_FIELDS)

PAIRED_COMPAT_CSV_REQUIRED_COLUMNS = [
    "utterance_id",
    "sample_id",
    "id",
    "emotion",
    "orig_label",
    "label",
    "label_id",
    "audio_path",
    "path",
    "filename",
    "source_group",
    "group_id",
    "duration",
    "transcript_final",
    "text_for_model",
    "text",
    "normalized_text",
    "split",
]


def export_paired_schema() -> dict[str, object]:
    return {
        "name": "vnemos_paired_mer",
        "source_type": "existing_paired_splits",
        "format": "jsonl",
        "required_fields": PAIRED_JSONL_REQUIRED_FIELDS,
        "field_descriptions": {
            "utterance_id": "Stable utterance-level id used across text, speech, calibration, and fusion.",
            "emotion": "Canonical MER5 emotion label.",
            "audio_path": "Project-relative path to processed mono 16kHz waveform.",
            "duration": "Audio duration in seconds.",
            "transcript_final": "Canonical transcript used for MER.",
            "text_for_model": "Final text field fed into the text branch. Must equal transcript_final.",
            "split": "One of: train, valid, test.",
        },
    }