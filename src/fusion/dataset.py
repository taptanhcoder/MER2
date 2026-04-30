from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from src.fusion.io import load_torch_artifact
from src.utils.paths import resolve_project_path


class FusionArtifactDataset(Dataset):
    def __init__(self, artifact_path: str | Path, project_root: str | Path = ".") -> None:
        self.artifact_path = resolve_project_path(artifact_path, start=project_root)
        self.artifact = load_torch_artifact(self.artifact_path)

        self.sample_ids = list(self.artifact["sample_id"])
        self.labels = self.artifact["label_id"].long()

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        metadata = self.artifact.get("metadata", {})
        return {
            "id": str(self.artifact["sample_id"][idx]),
            "label_id": int(self.artifact["label_id"][idx]),
            "text_logits_raw": self.artifact["text_logits_raw"][idx].float(),
            "speech_logits_raw": self.artifact["speech_logits_raw"][idx].float(),
            "text_probs_cal": self.artifact["text_probs_cal"][idx].float(),
            "speech_probs_cal": self.artifact["speech_probs_cal"][idx].float(),
            "text_embedding": self.artifact["text_embedding"][idx].float(),
            "speech_embedding": self.artifact["speech_embedding"][idx].float(),
            "text_tokens": self.artifact["text_tokens"][idx].float(),
            "speech_tokens": self.artifact["speech_tokens"][idx].float(),
            "text_token_mask": self.artifact["text_token_masks"][idx].long(),
            "speech_token_mask": self.artifact["speech_token_masks"][idx].long(),
            "reliability": self.artifact["reliability"][idx].float(),
            "text": str(metadata.get("text", [""])[idx]) if "text" in metadata else "",
            "raw_text": str(metadata.get("raw_text", [""])[idx]) if "raw_text" in metadata else "",
            "audio_path": str(metadata.get("audio_path", [""])[idx]) if "audio_path" in metadata else "",
            "group_id": str(metadata.get("group_id", [""])[idx]) if "group_id" in metadata else "",
            "duration": float(metadata.get("duration", [0.0])[idx]) if "duration" in metadata else 0.0,
        }


class FusionCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        ids = [row["id"] for row in batch]
        labels = torch.tensor([row["label_id"] for row in batch], dtype=torch.long)

        return {
            "ids": ids,
            "labels": labels,
            "text_logits_raw": torch.stack([row["text_logits_raw"] for row in batch], dim=0),
            "speech_logits_raw": torch.stack([row["speech_logits_raw"] for row in batch], dim=0),
            "text_probs_cal": torch.stack([row["text_probs_cal"] for row in batch], dim=0),
            "speech_probs_cal": torch.stack([row["speech_probs_cal"] for row in batch], dim=0),
            "text_embedding": torch.stack([row["text_embedding"] for row in batch], dim=0),
            "speech_embedding": torch.stack([row["speech_embedding"] for row in batch], dim=0),
            "text_tokens": torch.stack([row["text_tokens"] for row in batch], dim=0),
            "speech_tokens": torch.stack([row["speech_tokens"] for row in batch], dim=0),
            "text_token_mask": torch.stack([row["text_token_mask"] for row in batch], dim=0),
            "speech_token_mask": torch.stack([row["speech_token_mask"] for row in batch], dim=0),
            "reliability": torch.stack([row["reliability"] for row in batch], dim=0),
            "texts": [row["text"] for row in batch],
            "raw_texts": [row["raw_text"] for row in batch],
            "audio_paths": [row["audio_path"] for row in batch],
            "group_ids": [row["group_id"] for row in batch],
            "durations": [row["duration"] for row in batch],
        }


def build_fusion_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    runtime_cfg = config.get("runtime", {})
    train_cfg = config["train"]
    project_root = config.get("project", {}).get("root", ".")

    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(runtime_cfg.get("num_workers", 0))
    pin_memory = bool(runtime_cfg.get("pin_memory", False))
    persistent_workers = bool(runtime_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = runtime_cfg.get("prefetch_factor", None)

    train_ds = FusionArtifactDataset(dataset_cfg["fusion_artifacts"]["train"], project_root=project_root)
    valid_ds = FusionArtifactDataset(dataset_cfg["fusion_artifacts"]["valid"], project_root=project_root)
    test_ds = FusionArtifactDataset(dataset_cfg["fusion_artifacts"]["test"], project_root=project_root)

    collator = FusionCollator()

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, valid_loader, test_loader