from __future__ import annotations

import json
import math
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
from src.text.preprocess import prepare_model_text
from src.text.truncation import (
    encode_text_with_strategy,
    normalize_truncation_strategy,
    pad_encoded_rows,
    uses_custom_truncation,
)
from src.utils.paths import resolve_project_path


@dataclass
class TextSample:
    sample_id: str
    raw_text: str
    text: str
    label: str
    label_id: int
    audio_path: str
    group_id: str


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        dataset_config: dict[str, Any],
        preprocessing_config: dict[str, Any],
        project_root: str | Path = ".",
    ) -> None:
        self.csv_path = resolve_project_path(csv_path, start=project_root)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Text manifest not found: {self.csv_path}")

        self.dataset_config = dataset_config
        self.preprocessing_config = preprocessing_config
        self.frame = pd.read_csv(self.csv_path)

        id_col = dataset_config.get("id_col", "sample_id")
        text_col = dataset_config.get("text_col", "text")
        raw_text_col = dataset_config.get("raw_text_col", text_col)
        normalized_text_col = dataset_config.get("normalized_text_col", "normalized_text")
        audio_path_col = dataset_config.get("audio_path_col", "audio_path")
        group_id_col = dataset_config.get("group_id_col", "group_id")
        label_col = dataset_config.get("label_col", "label")
        label_id_col = dataset_config.get("label_id_col", "label_id")
        use_normalized_text = bool(dataset_config.get("use_normalized_text", False))
        num_classes = int(dataset_config.get("num_classes", 5))

        if use_normalized_text and normalized_text_col in self.frame.columns:
            source_text_col = normalized_text_col
        else:
            source_text_col = text_col

        required_cols = [id_col, source_text_col, raw_text_col, label_col]
        missing_required = [col for col in required_cols if col not in self.frame.columns]
        if missing_required:
            raise ValueError(
                f"Missing required columns in {self.csv_path}: {missing_required}. "
                f"Available columns: {self.frame.columns.tolist()}"
            )

        has_label_id = label_id_col in self.frame.columns

        self.samples: list[TextSample] = []
        for _, row in self.frame.iterrows():
            raw_text = str(row[raw_text_col])
            source_text = str(row[source_text_col])

            model_text = prepare_model_text(
                text=source_text,
                normalize_whitespace=bool(
                    preprocessing_config.get("normalize_whitespace", True)
                ),
                lowercase=bool(preprocessing_config.get("lowercase", False)),
                word_segment=bool(preprocessing_config.get("word_segment", False)),
                preserve_spoken_style=bool(
                    preprocessing_config.get("preserve_spoken_style", True)
                ),
            )

            label_name = str(row[label_col])
            if has_label_id and pd.notna(row[label_id_col]):
                label_id = int(row[label_id_col])
            else:
                label_id = int(label_to_id(label_name))

            audio_path = (
                str(row[audio_path_col])
                if audio_path_col in self.frame.columns
                else ""
            )
            group_id = (
                str(row[group_id_col])
                if group_id_col in self.frame.columns
                else ""
            )

            self.samples.append(
                TextSample(
                    sample_id=str(row[id_col]),
                    raw_text=raw_text,
                    text=model_text,
                    label=label_name,
                    label_id=label_id,
                    audio_path=audio_path,
                    group_id=group_id,
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

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        return {
            "id": sample.sample_id,
            "raw_text": sample.raw_text,
            "text": sample.text,
            "label": sample.label,
            "label_id": sample.label_id,
            "audio_path": sample.audio_path,
            "group_id": sample.group_id,
        }


class TextCollator:
    def __init__(
        self,
        tokenizer,
        tokenizer_config: dict[str, Any],
        training: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.tokenizer_config = tokenizer_config
        self.training = bool(training)

    def _encode_with_default_tokenizer(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=self.tokenizer_config.get("padding", True),
            truncation=self.tokenizer_config.get("truncation", True),
            max_length=int(self.tokenizer_config.get("max_length", 128)),
            return_tensors="pt",
        )

    def _encode_with_length_aware_strategy(
        self,
        texts: list[str],
    ) -> tuple[dict[str, torch.Tensor], list[int], list[bool]]:
        max_length = int(self.tokenizer_config.get("max_length", 128))
        strategy = normalize_truncation_strategy(
            self.tokenizer_config.get("truncation_strategy", "first")
        )
        padding = self.tokenizer_config.get("padding", True)

        encoded_rows: list[dict[str, list[int]]] = []
        token_lengths: list[int] = []
        truncated_flags: list[bool] = []

        for text in texts:
            input_ids, attention_mask, original_token_len, was_truncated = (
                encode_text_with_strategy(
                    tokenizer=self.tokenizer,
                    text=text,
                    max_length=max_length,
                    strategy=strategy,
                    training=self.training,
                )
            )

            encoded_rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }
            )
            token_lengths.append(int(original_token_len))
            truncated_flags.append(bool(was_truncated))

        padded = pad_encoded_rows(
            tokenizer=self.tokenizer,
            encoded_rows=encoded_rows,
            padding=padding,
            max_length=max_length,
        )
        return padded, token_lengths, truncated_flags

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [row["text"] for row in batch]
        raw_texts = [row["raw_text"] for row in batch]
        ids = [row["id"] for row in batch]
        audio_paths = [row.get("audio_path", "") for row in batch]
        group_ids = [row.get("group_id", "") for row in batch]
        labels = torch.tensor([row["label_id"] for row in batch], dtype=torch.long)

        strategy = normalize_truncation_strategy(
            self.tokenizer_config.get("truncation_strategy", "first")
        )

        if uses_custom_truncation(self.tokenizer_config):
            encoded, token_lengths, truncated_flags = (
                self._encode_with_length_aware_strategy(texts)
            )
        else:
            encoded = self._encode_with_default_tokenizer(texts)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                token_lengths = [int(mask.sum().item()) for mask in attention_mask]
            else:
                token_lengths = [int(encoded["input_ids"].shape[1]) for _ in texts]
            truncated_flags = [False for _ in texts]

        result = {
            "ids": ids,
            "texts": texts,
            "raw_texts": raw_texts,
            "audio_paths": audio_paths,
            "group_ids": group_ids,
            "labels": labels,
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded.get("attention_mask"),
            "token_lengths": token_lengths,
            "truncated_flags": truncated_flags,
            "truncation_strategy": strategy,
        }

        if "token_type_ids" in encoded:
            result["token_type_ids"] = encoded["token_type_ids"]

        return result


def build_text_dataloaders(
    config: dict[str, Any],
    tokenizer,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_config = config["dataset"]
    preprocessing_config = config.get("preprocessing", {})
    tokenizer_config = config["tokenizer"]
    train_config = config["train"]
    runtime_config = config.get("runtime", {})
    project_root = config.get("project", {}).get("root", ".")

    batch_size = int(train_config.get("batch_size", 16))
    num_workers = int(runtime_config.get("num_workers", 0))
    pin_memory = bool(runtime_config.get("pin_memory", False))
    persistent_workers = bool(runtime_config.get("persistent_workers", num_workers > 0))
    prefetch_factor = runtime_config.get("prefetch_factor", None)

    train_ds = TextClassificationDataset(
        csv_path=dataset_config["train_csv"],
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        project_root=project_root,
    )
    valid_ds = TextClassificationDataset(
        csv_path=dataset_config["valid_csv"],
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        project_root=project_root,
    )
    test_ds = TextClassificationDataset(
        csv_path=dataset_config["test_csv"],
        dataset_config=dataset_config,
        preprocessing_config=preprocessing_config,
        project_root=project_root,
    )

    train_collator = TextCollator(
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        training=True,
    )
    eval_collator = TextCollator(
        tokenizer=tokenizer,
        tokenizer_config=tokenizer_config,
        training=False,
    )

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=train_collator,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_ds,
        shuffle=False,
        collate_fn=eval_collator,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        collate_fn=eval_collator,
        **loader_kwargs,
    )

    return train_loader, valid_loader, test_loader


class TextTrainer(BaseTrainer):
    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        prepared = {
            "ids": batch["ids"],
            "labels": batch["labels"].to(self.device),
            "input_ids": batch["input_ids"].to(self.device),
            "prediction_metadata": {
                "texts": [str(x) for x in batch.get("texts", [])],
                "raw_texts": [str(x) for x in batch.get("raw_texts", [])],
                "audio_paths": [str(x) for x in batch.get("audio_paths", [])],
                "group_ids": [str(x) for x in batch.get("group_ids", [])],
                "token_lengths": [
                    int(x) for x in batch.get("token_lengths", [])
                ],
                "truncated_flags": [
                    bool(x) for x in batch.get("truncated_flags", [])
                ],
                "truncation_strategy": [
                    str(batch.get("truncation_strategy", "first"))
                    for _ in batch.get("ids", [])
                ],
            },
        }

        if batch.get("attention_mask") is not None:
            prepared["attention_mask"] = batch["attention_mask"].to(self.device)

        if batch.get("token_type_ids") is not None:
            prepared["token_type_ids"] = batch["token_type_ids"].to(self.device)

        return prepared

    def forward_step(self, batch: dict[str, Any]) -> ClassifierOutput:
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids"),
            labels=batch.get("labels"),
        )


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


def epoch_result_to_dataframe(
    result: EpochResult,
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    texts = result.predictions.metadata.get("texts", [])
    raw_texts = result.predictions.metadata.get("raw_texts", [])
    audio_paths = result.predictions.metadata.get("audio_paths", [])
    group_ids = result.predictions.metadata.get("group_ids", [])
    token_lengths = result.predictions.metadata.get("token_lengths", [])
    truncated_flags = result.predictions.metadata.get("truncated_flags", [])
    truncation_strategy = result.predictions.metadata.get("truncation_strategy", [])

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

        if idx < len(texts):
            row["text"] = texts[idx]
        if idx < len(raw_texts):
            row["raw_text"] = raw_texts[idx]
        if idx < len(audio_paths):
            row["audio_path"] = audio_paths[idx]
        if idx < len(group_ids):
            row["group_id"] = group_ids[idx]
        if idx < len(token_lengths):
            row["token_len"] = int(token_lengths[idx])
        if idx < len(truncated_flags):
            row["was_truncated"] = bool(truncated_flags[idx])
        if idx < len(truncation_strategy):
            row["truncation_strategy"] = str(truncation_strategy[idx])

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