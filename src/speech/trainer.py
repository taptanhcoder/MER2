from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.common.base_trainer import BaseTrainer
from src.common.interfaces import ClassifierOutput, EpochResult
from src.common.losses import balanced_class_weights_from_counts
from src.data.label_space import label_to_id
from src.speech.audio_io import load_audio
from src.speech.augment import build_waveform_augment
from src.utils.paths import resolve_project_path


@dataclass
class SpeechSample:
    sample_id: str
    audio_path: str
    resolved_audio_path: str
    label: str
    label_id: int
    group_id: str
    duration: float
    text: str
    raw_text: str
    split: str


def _crop_or_pad_waveform(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    if waveform.dim() == 2 and waveform.size(0) == 1:
        waveform = waveform.squeeze(0)

    num_samples = int(waveform.shape[-1])
    if num_samples == target_num_samples:
        return waveform

    if num_samples > target_num_samples:
        return waveform[:target_num_samples]

    pad_amount = target_num_samples - num_samples
    return torch.nn.functional.pad(waveform, (0, pad_amount))


def _random_segment(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    if waveform.dim() == 2 and waveform.size(0) == 1:
        waveform = waveform.squeeze(0)

    num_samples = int(waveform.shape[-1])
    if num_samples <= target_num_samples:
        return _crop_or_pad_waveform(waveform, target_num_samples)

    max_start = num_samples - target_num_samples
    start = random.randint(0, max_start)
    return waveform[start : start + target_num_samples]


def _deterministic_segments(
    waveform: torch.Tensor,
    target_num_samples: int,
    num_views: int,
) -> list[torch.Tensor]:
    if waveform.dim() == 2 and waveform.size(0) == 1:
        waveform = waveform.squeeze(0)

    num_samples = int(waveform.shape[-1])
    if num_samples <= target_num_samples:
        return [_crop_or_pad_waveform(waveform, target_num_samples)]

    num_views = max(1, int(num_views))
    max_start = num_samples - target_num_samples
    if num_views == 1:
        start = max_start // 2
        return [waveform[start : start + target_num_samples]]

    starts = torch.linspace(0, max_start, steps=num_views).round().long().tolist()
    return [waveform[start : start + target_num_samples] for start in starts]


def _entropy(prob_row: list[float]) -> float:
    eps = 1e-12
    return float(-sum(p * math.log(max(p, eps)) for p in prob_row))


def _top2_margin(prob_row: list[float]) -> float:
    if not prob_row:
        return 0.0
    top2 = sorted(prob_row, reverse=True)[:2]
    if len(top2) == 1:
        return float(top2[0])
    return float(top2[0] - top2[1])


class SpeechClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        dataset_config: dict[str, Any],
        audio_config: dict[str, Any],
        augment_config: dict[str, Any] | None = None,
        split: str = "train",
        project_root: str | Path = ".",
        deterministic_export: bool = False,
        export_num_views: int = 1,
    ) -> None:
        self.csv_path = resolve_project_path(csv_path, start=project_root)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Speech manifest not found: {self.csv_path}")

        self.dataset_config = dataset_config
        self.audio_config = audio_config
        self.split = split
        self.deterministic_export = bool(deterministic_export)
        self.export_num_views = max(1, int(export_num_views))
        self.frame = pd.read_csv(self.csv_path)

        id_col = dataset_config.get("id_col", "sample_id")
        audio_path_col = dataset_config.get("audio_path_col", "audio_path")
        duration_col = dataset_config.get("duration_col", "duration")
        group_id_col = dataset_config.get("group_id_col", "group_id")
        text_col = dataset_config.get("text_col", "text")
        raw_text_col = dataset_config.get("raw_text_col", "transcript_final")
        label_col = dataset_config.get("label_col", "label")
        label_id_col = dataset_config.get("label_id_col", "label_id")
        num_classes = int(dataset_config.get("num_classes", 5))

        required_cols = [id_col, audio_path_col, label_col]
        missing_required = [c for c in required_cols if c not in self.frame.columns]
        if missing_required:
            raise ValueError(
                f"Missing required columns in {self.csv_path}: {missing_required}. "
                f"Available columns: {self.frame.columns.tolist()}"
            )

        self.target_sample_rate = int(
            audio_config.get("sample_rate", audio_config.get("target_sample_rate", 16000))
        )
        self.mono = bool(audio_config.get("mono", True))
        self.max_duration_sec = float(audio_config.get("max_duration_sec", 8.0))
        self.num_train_views = max(1, int(audio_config.get("num_train_views", 1)))
        self.num_eval_views = max(1, int(audio_config.get("num_eval_views", 3)))
        self.target_num_samples = max(1, int(self.max_duration_sec * self.target_sample_rate))

        self.waveform_augment = None
        if split == "train" and not self.deterministic_export:
            self.waveform_augment = build_waveform_augment(augment_config)

        self.samples: list[SpeechSample] = []
        for _, row in self.frame.iterrows():
            sample_id = str(row[id_col])
            audio_path = str(row[audio_path_col])
            resolved_audio_path = str(resolve_project_path(audio_path, start=project_root))
            label_name = str(row[label_col])

            if label_id_col in self.frame.columns and pd.notna(row[label_id_col]):
                label_id = int(row[label_id_col])
            else:
                label_id = int(label_to_id(label_name))

            duration = float(row[duration_col]) if duration_col in self.frame.columns else 0.0
            group_id = str(row[group_id_col]) if group_id_col in self.frame.columns else ""
            text = str(row[text_col]) if text_col in self.frame.columns else ""
            raw_text = str(row[raw_text_col]) if raw_text_col in self.frame.columns else text

            self.samples.append(
                SpeechSample(
                    sample_id=sample_id,
                    audio_path=audio_path,
                    resolved_audio_path=resolved_audio_path,
                    label=label_name,
                    label_id=label_id,
                    group_id=group_id,
                    duration=duration,
                    text=text,
                    raw_text=raw_text,
                    split=split,
                )
            )

        self.class_counts = [0 for _ in range(num_classes)]
        for sample in self.samples:
            if 0 <= sample.label_id < num_classes:
                self.class_counts[sample.label_id] += 1

        self.class_weights = (
            balanced_class_weights_from_counts(self.class_counts)
            if self.class_counts
            else []
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_segments(self, sample: SpeechSample) -> list[torch.Tensor]:
        waveform, _ = load_audio(
            path=sample.resolved_audio_path,
            target_sample_rate=self.target_sample_rate,
            mono=self.mono,
        )

        if waveform.dim() == 2 and waveform.size(0) == 1:
            waveform = waveform.squeeze(0)

        waveform = waveform.float()

        if self.deterministic_export:
            return _deterministic_segments(
                waveform=waveform,
                target_num_samples=self.target_num_samples,
                num_views=self.export_num_views,
            )

        if self.split == "train":
            segments = [
                _random_segment(waveform, self.target_num_samples)
                for _ in range(self.num_train_views)
            ]
            if self.waveform_augment is not None:
                segments = [self.waveform_augment(segment) for segment in segments]
            return segments

        return _deterministic_segments(
            waveform=waveform,
            target_num_samples=self.target_num_samples,
            num_views=self.num_eval_views,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        segments = self._load_segments(sample)

        return {
            "id": sample.sample_id,
            "segments": segments,
            "audio_path": sample.audio_path,
            "group_id": sample.group_id,
            "duration": sample.duration,
            "text": sample.text,
            "raw_text": sample.raw_text,
            "label": sample.label,
            "label_id": sample.label_id,
            "split": sample.split,
        }


class SpeechCollator:
    def __init__(self, processor, sample_rate: int) -> None:
        self.processor = processor
        self.sample_rate = int(sample_rate)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        ids = [row["id"] for row in batch]
        labels = torch.tensor([row["label_id"] for row in batch], dtype=torch.long)

        audio_paths = [row.get("audio_path", "") for row in batch]
        group_ids = [row.get("group_id", "") for row in batch]
        durations = [float(row.get("duration", 0.0)) for row in batch]
        texts = [row.get("text", "") for row in batch]
        raw_texts = [row.get("raw_text", "") for row in batch]

        flat_segments: list[torch.Tensor] = []
        segment_parent_indices: list[int] = []
        for sample_idx, row in enumerate(batch):
            segments = row["segments"]
            for segment in segments:
                flat_segments.append(segment.detach().cpu().float())
                segment_parent_indices.append(sample_idx)

        encoded = self.processor(
            [segment.numpy() for segment in flat_segments],
            sampling_rate=self.sample_rate,
            padding=True,
            return_tensors="pt",
        )

        result = {
            "ids": ids,
            "labels": labels,
            "audio_paths": audio_paths,
            "group_ids": group_ids,
            "durations": durations,
            "texts": texts,
            "raw_texts": raw_texts,
            "segment_parent_indices": torch.tensor(segment_parent_indices, dtype=torch.long),
            "input_values": encoded["input_values"],
            "attention_mask": encoded.get("attention_mask"),
        }
        return result


def _build_loader(
    dataset: SpeechClassificationDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    collator: SpeechCollator,
    shuffle: bool,
) -> DataLoader:
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collator,
        "shuffle": shuffle,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(dataset, **loader_kwargs)


def build_speech_dataloaders(
    config: dict[str, Any],
    processor,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_config = config["dataset"]
    audio_config = config.get("audio", config.get("audio_config", {}))
    augment_config = config.get("augment", {})
    train_config = config["train"]
    runtime_config = config.get("runtime", {})
    project_root = config.get("project", {}).get("root", ".")

    batch_size = int(train_config.get("batch_size", 8))
    num_workers = int(runtime_config.get("num_workers", 0))
    pin_memory = bool(runtime_config.get("pin_memory", False))
    persistent_workers = bool(runtime_config.get("persistent_workers", num_workers > 0))
    prefetch_factor = runtime_config.get("prefetch_factor", None)

    sample_rate = int(audio_config.get("sample_rate", audio_config.get("target_sample_rate", 16000)))

    train_ds = SpeechClassificationDataset(
        csv_path=dataset_config["train_csv"],
        dataset_config=dataset_config,
        audio_config=audio_config,
        augment_config=augment_config,
        split="train",
        project_root=project_root,
    )
    valid_ds = SpeechClassificationDataset(
        csv_path=dataset_config["valid_csv"],
        dataset_config=dataset_config,
        audio_config=audio_config,
        augment_config=None,
        split="valid",
        project_root=project_root,
    )
    test_ds = SpeechClassificationDataset(
        csv_path=dataset_config["test_csv"],
        dataset_config=dataset_config,
        audio_config=audio_config,
        augment_config=None,
        split="test",
        project_root=project_root,
    )

    collator = SpeechCollator(processor=processor, sample_rate=sample_rate)
    train_loader = _build_loader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collator=collator,
        shuffle=True,
    )
    valid_loader = _build_loader(
        dataset=valid_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collator=collator,
        shuffle=False,
    )
    test_loader = _build_loader(
        dataset=test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collator=collator,
        shuffle=False,
    )
    return train_loader, valid_loader, test_loader


def build_speech_export_dataloaders(
    config: dict[str, Any],
    processor,
    export_num_views: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Deterministic dataloaders for fusion feature export.

    This mode is intentionally separate from training/evaluation loaders:
    - no random crop
    - no augmentation
    - single canonical view by default
    """
    dataset_config = config["dataset"]
    audio_config = config.get("audio", config.get("audio_config", {}))
    train_config = config["train"]
    runtime_config = config.get("runtime", {})
    project_root = config.get("project", {}).get("root", ".")

    batch_size = int(train_config.get("batch_size", 8))
    num_workers = int(runtime_config.get("num_workers", 0))
    pin_memory = bool(runtime_config.get("pin_memory", False))
    persistent_workers = bool(runtime_config.get("persistent_workers", num_workers > 0))
    prefetch_factor = runtime_config.get("prefetch_factor", None)

    sample_rate = int(audio_config.get("sample_rate", audio_config.get("target_sample_rate", 16000)))

    train_ds = SpeechClassificationDataset(
        csv_path=dataset_config["train_csv"],
        dataset_config=dataset_config,
        audio_config=audio_config,
        augment_config=None,
        split="train",
        project_root=project_root,
        deterministic_export=True,
        export_num_views=export_num_views,
    )
    valid_ds = SpeechClassificationDataset(
        csv_path=dataset_config["valid_csv"],
        dataset_config=dataset_config,
        audio_config=audio_config,
        augment_config=None,
        split="valid",
        project_root=project_root,
        deterministic_export=True,
        export_num_views=export_num_views,
    )
    test_ds = SpeechClassificationDataset(
        csv_path=dataset_config["test_csv"],
        dataset_config=dataset_config,
        audio_config=audio_config,
        augment_config=None,
        split="test",
        project_root=project_root,
        deterministic_export=True,
        export_num_views=export_num_views,
    )

    collator = SpeechCollator(processor=processor, sample_rate=sample_rate)
    train_loader = _build_loader(
        dataset=train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collator=collator,
        shuffle=False,
    )
    valid_loader = _build_loader(
        dataset=valid_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collator=collator,
        shuffle=False,
    )
    test_loader = _build_loader(
        dataset=test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        collator=collator,
        shuffle=False,
    )
    return train_loader, valid_loader, test_loader


class SpeechTrainer(BaseTrainer):
    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        prepared = {
            "ids": batch["ids"],
            "labels": batch["labels"].to(self.device),
            "input_values": batch["input_values"].to(self.device),
            "segment_parent_indices": batch["segment_parent_indices"].to(self.device),
            "prediction_metadata": {
                "audio_paths": [str(x) for x in batch.get("audio_paths", [])],
                "group_ids": [str(x) for x in batch.get("group_ids", [])],
                "durations": [float(x) for x in batch.get("durations", [])],
                "texts": [str(x) for x in batch.get("texts", [])],
                "raw_texts": [str(x) for x in batch.get("raw_texts", [])],
            },
        }
        if batch.get("attention_mask") is not None:
            prepared["attention_mask"] = batch["attention_mask"].to(self.device)
        return prepared

    def _aggregate_segment_outputs(
        self,
        logits: torch.Tensor,
        embeddings: torch.Tensor | None,
        parent_indices: torch.Tensor,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_classes = int(logits.shape[-1])

        aggregated_logits = torch.zeros(
            num_samples,
            num_classes,
            device=logits.device,
            dtype=logits.dtype,
        )
        counts = torch.zeros(
            num_samples,
            1,
            device=logits.device,
            dtype=logits.dtype,
        )
        ones = torch.ones(
            parent_indices.shape[0],
            1,
            device=logits.device,
            dtype=logits.dtype,
        )

        aggregated_logits.index_add_(0, parent_indices, logits)
        counts.index_add_(0, parent_indices, ones)
        aggregated_logits = aggregated_logits / counts.clamp_min(1.0)

        aggregated_embeddings = None
        if embeddings is not None:
            embedding_dim = int(embeddings.shape[-1])
            aggregated_embeddings = torch.zeros(
                num_samples,
                embedding_dim,
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
            aggregated_embeddings.index_add_(0, parent_indices, embeddings)
            aggregated_embeddings = aggregated_embeddings / counts.clamp_min(1.0)

        return aggregated_logits, aggregated_embeddings

    def forward_step(self, batch: dict[str, Any]) -> ClassifierOutput:
        segment_output = self.model(
            input_values=batch["input_values"],
            attention_mask=batch.get("attention_mask"),
            labels=None,
        )

        aggregated_logits, aggregated_embeddings = self._aggregate_segment_outputs(
            logits=segment_output.logits,
            embeddings=segment_output.embeddings,
            parent_indices=batch["segment_parent_indices"],
            num_samples=int(batch["labels"].shape[0]),
        )

        probs = torch.softmax(aggregated_logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        return ClassifierOutput(
            loss=None,
            logits=aggregated_logits,
            probs=probs,
            preds=preds,
            labels=batch["labels"],
            embeddings=aggregated_embeddings,
        )


def epoch_result_to_dataframe(
    result: EpochResult,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    audio_paths = result.predictions.metadata.get("audio_paths", [])
    group_ids = result.predictions.metadata.get("group_ids", [])
    durations = result.predictions.metadata.get("durations", [])
    texts = result.predictions.metadata.get("texts", [])
    raw_texts = result.predictions.metadata.get("raw_texts", [])

    for idx, (sample_id, label_id, pred_id, prob_row, logit_row) in enumerate(
        zip(
            result.predictions.ids,
            result.predictions.labels,
            result.predictions.preds,
            result.predictions.probs,
            result.predictions.logits,
        )
    ):
        row = {
            "sample_id": sample_id,
            "label_id": int(label_id),
            "pred_id": int(pred_id),
            "confidence": float(max(prob_row)),
            "entropy": _entropy(prob_row),
            "top2_margin": _top2_margin(prob_row),
            "probs": json.dumps(prob_row, ensure_ascii=False),
            "logits": json.dumps(logit_row, ensure_ascii=False),
            "split": result.split,
        }

        if idx < len(audio_paths):
            row["audio_path"] = audio_paths[idx]
        if idx < len(group_ids):
            row["group_id"] = group_ids[idx]
        if idx < len(durations):
            row["duration"] = float(durations[idx])
        if idx < len(texts):
            row["text"] = texts[idx]
        if idx < len(raw_texts):
            row["raw_text"] = raw_texts[idx]

        if label_names is not None:
            row["label"] = label_names[int(label_id)]
            row["pred"] = label_names[int(pred_id)]

        rows.append(row)

    return pd.DataFrame(rows)


def save_epoch_predictions(
    result: EpochResult,
    output_path: str | Path,
    label_names: list[str] | None = None,
) -> None:
    df = epoch_result_to_dataframe(result, label_names=label_names)
    file_path = Path(output_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)