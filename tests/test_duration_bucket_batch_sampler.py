from __future__ import annotations

from dataclasses import dataclass

from src.speech.trainer import DurationBucketBatchSampler


@dataclass
class DummySample:
    duration_sec: float


class DummyDataset:
    def __init__(self) -> None:
        self.samples = [
            DummySample(2.0),
            DummySample(2.2),
            DummySample(7.5),
            DummySample(7.9),
            DummySample(10.0),
            DummySample(10.3),
        ]

    def __len__(self) -> int:
        return len(self.samples)


def test_duration_bucket_batch_sampler_yields_full_coverage() -> None:
    dataset = DummyDataset()
    sampler = DurationBucketBatchSampler(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
        drop_last=False,
        bucket_size_multiplier=3,
    )

    batches = list(iter(sampler))
    flattened = [idx for batch in batches for idx in batch]

    assert len(batches) == 3
    assert sorted(flattened) == list(range(len(dataset)))