import torch

from src.speech.trainer import SpeechCollator


class DummyProcessor:
    sampling_rate = 16000

    def __call__(
        self,
        waveforms,
        sampling_rate,
        padding,
        return_tensors,
        return_attention_mask=True,
    ):
        assert sampling_rate == 16000
        max_length = max(len(w) for w in waveforms)
        batch_size = len(waveforms)

        input_values = torch.zeros(batch_size, max_length, dtype=torch.float32)
        attention_mask = torch.zeros(batch_size, max_length, dtype=torch.long)

        for i, waveform in enumerate(waveforms):
            waveform_tensor = torch.tensor(waveform, dtype=torch.float32)
            input_values[i, : waveform_tensor.numel()] = waveform_tensor
            attention_mask[i, : waveform_tensor.numel()] = 1

        outputs = {"input_values": input_values}
        if return_attention_mask:
            outputs["attention_mask"] = attention_mask
        return outputs


def test_speech_collator_pads_variable_length_waveforms():
    batch = [
        {
            "sample_id": "a",
            "path": "a.wav",
            "filename": "a.wav",
            "group_id": "g1",
            "label": "anger",
            "label_id": 0,
            "waveform": torch.ones(8),
            "sampling_rate": 16000,
            "raw_length": 8,
            "duration_sec": 8 / 16000.0,
        },
        {
            "sample_id": "b",
            "path": "b.wav",
            "filename": "b.wav",
            "group_id": "g2",
            "label": "sadness",
            "label_id": 3,
            "waveform": torch.ones(5),
            "sampling_rate": 16000,
            "raw_length": 5,
            "duration_sec": 5 / 16000.0,
        },
    ]

    collator = SpeechCollator(processor=DummyProcessor(), target_sample_rate=16000)
    result = collator(batch)

    assert result["input_values"].shape == (2, 8)
    assert result["attention_mask"].shape == (2, 8)
    assert result["labels"].tolist() == [0, 3]
    assert result["raw_lengths"].tolist() == [8, 5]
    assert result["paths"] == ["a.wav", "b.wav"]