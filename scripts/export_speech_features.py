from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.checkpoint import load_checkpoint
from src.common.registry import MODEL_REGISTRY
from src.fusion.export_utils import (
    compress_sequence_to_token_bank,
    find_best_checkpoint,
    load_checkpoint_state_flexibly,
    load_run_or_explicit_config,
    normalize_legacy_model_keys,
    patch_speech_dataset_paths_if_legacy,
)
from src.fusion.io import validate_expert_export_artifact
from src.speech.models.hubert import HuBERTClassifier  # noqa: F401
from src.speech.processors import build_speech_processor
from src.speech.trainer import build_speech_export_dataloaders


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export speech expert features for fusion.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Speech run directory.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional explicit config path. Useful when run artifact uses legacy dataset paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fusion_exports/speech"),
        help="Output directory for exported speech features.",
    )
    parser.add_argument(
        "--compact-frames",
        type=int,
        default=16,
        help="Number of compact speech tokens/frames to export for fusion.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cpu, cuda, auto")
    return parser


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_name = str(config["model"]["name"])
    model_cls = MODEL_REGISTRY.get(model_name)
    return model_cls(config)


def _compute_feature_attention_mask(
    model,
    attention_mask: torch.Tensor | None,
    feature_length: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones(batch_size, feature_length, dtype=torch.long, device=device)

    encoder = model.encoder
    if hasattr(encoder, "_get_feature_vector_attention_mask"):
        return encoder._get_feature_vector_attention_mask(feature_length, attention_mask)

    return torch.ones(batch_size, feature_length, dtype=torch.long, device=device)


def export_split(
    model,
    loader,
    device: torch.device,
    output_path: Path,
    compact_frames: int,
) -> None:
    sample_ids: list[str] = []
    label_ids: list[int] = []
    logits_rows: list[torch.Tensor] = []
    probs_rows: list[torch.Tensor] = []
    pooled_rows: list[torch.Tensor] = []
    compact_token_rows: list[torch.Tensor] = []
    compact_mask_rows: list[torch.Tensor] = []

    metadata = {
        "text": [],
        "raw_text": [],
        "audio_path": [],
        "group_id": [],
        "duration": [],
    }

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            if input_values.shape[0] != len(batch["ids"]):
                raise RuntimeError(
                    "Speech fusion export expects deterministic single-view input, "
                    "but received multiple segments per sample."
                )

            outputs = model.encoder(
                input_values=input_values,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            feature_mask = _compute_feature_attention_mask(
                model=model,
                attention_mask=attention_mask,
                feature_length=int(hidden_states.shape[1]),
                batch_size=int(hidden_states.shape[0]),
                device=device,
            )

            pooled = model.pooling(hidden_states, attention_mask=feature_mask)
            logits = model.classifier(model.dropout(pooled))
            probs = torch.softmax(logits, dim=-1)

            batch_ids = [str(x) for x in batch["ids"]]
            batch_labels = batch["labels"].tolist()
            batch_texts = [str(x) for x in batch.get("texts", [])]
            batch_raw_texts = [str(x) for x in batch.get("raw_texts", [])]
            batch_audio_paths = [str(x) for x in batch.get("audio_paths", [])]
            batch_group_ids = [str(x) for x in batch.get("group_ids", [])]
            batch_durations = [float(x) for x in batch.get("durations", [])]

            for i in range(hidden_states.shape[0]):
                compact_tokens_i, compact_mask_i = compress_sequence_to_token_bank(
                    sequence=hidden_states[i],
                    mask=feature_mask[i],
                    target_len=compact_frames,
                )

                sample_ids.append(batch_ids[i])
                label_ids.append(int(batch_labels[i]))
                logits_rows.append(logits[i].detach().cpu().float())
                probs_rows.append(probs[i].detach().cpu().float())
                pooled_rows.append(pooled[i].detach().cpu().float())
                compact_token_rows.append(compact_tokens_i.detach().cpu().float())
                compact_mask_rows.append(compact_mask_i.detach().cpu().long())

                metadata["text"].append(batch_texts[i] if i < len(batch_texts) else "")
                metadata["raw_text"].append(batch_raw_texts[i] if i < len(batch_raw_texts) else "")
                metadata["audio_path"].append(batch_audio_paths[i] if i < len(batch_audio_paths) else "")
                metadata["group_id"].append(batch_group_ids[i] if i < len(batch_group_ids) else "")
                metadata["duration"].append(batch_durations[i] if i < len(batch_durations) else 0.0)

    artifact = {
        "sample_id": sample_ids,
        "label_id": torch.tensor(label_ids, dtype=torch.long),
        "logits": torch.stack(logits_rows, dim=0),
        "probs": torch.stack(probs_rows, dim=0),
        "pooled_embedding": torch.stack(pooled_rows, dim=0),
        "compact_tokens": torch.stack(compact_token_rows, dim=0),
        "compact_token_masks": torch.stack(compact_mask_rows, dim=0),
        "metadata": metadata,
    }
    validate_expert_export_artifact(artifact, "speech_export")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)


def main() -> int:
    args = build_parser().parse_args()

    run_dir = args.run_dir.resolve()
    config = load_run_or_explicit_config(
        run_dir=run_dir,
        explicit_config_path=args.config,
        project_root=PROJECT_ROOT,
    )
    config, path_patched = patch_speech_dataset_paths_if_legacy(config, PROJECT_ROOT)
    config, model_patched = normalize_legacy_model_keys(config)

    if path_patched:
        print("[INFO] Patched legacy speech dataset paths to current canonical VNEMOS CSVs.")
    if model_patched:
        print("[INFO] Patched legacy speech model registry key to current codebase naming.")

    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    processor = build_speech_processor(config["model"])
    train_loader, valid_loader, test_loader = build_speech_export_dataloaders(
        config=config,
        processor=processor,
        export_num_views=1,
    )

    model = build_model(config)
    ckpt_path = find_best_checkpoint(run_dir)
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    notes = load_checkpoint_state_flexibly(model, ckpt)
    for note in notes:
        print(f"[INFO] Adapted legacy checkpoint key: {note}")
    model.to(device)

    export_split(model, train_loader, device, args.output_dir / "train.pt", args.compact_frames)
    export_split(model, valid_loader, device, args.output_dir / "valid.pt", args.compact_frames)
    export_split(model, test_loader, device, args.output_dir / "test.pt", args.compact_frames)

    print(f"[OK] Exported speech features to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())