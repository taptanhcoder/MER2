from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

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
    patch_text_dataset_paths_if_legacy,
)
from src.fusion.io import validate_expert_export_artifact
from src.text.models.phobert import PhoBERTClassifier  # noqa: F401
from src.text.tokenizers import build_text_tokenizer
from src.text.trainer import build_text_dataloaders


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export text expert features for fusion.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Text run directory.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional explicit config path. Useful when run artifact uses legacy dataset paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fusion_exports/text"),
        help="Output directory for exported text features.",
    )
    parser.add_argument(
        "--compact-tokens",
        type=int,
        default=16,
        help="Number of compact text tokens to export for fusion.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="cpu, cuda, auto")
    return parser


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    model_name = str(config["model"]["name"])
    model_cls = MODEL_REGISTRY.get(model_name)
    return model_cls(config)


def run_classifier_head(model, pooled: torch.Tensor) -> torch.Tensor:
    """
    Export-time classifier head call must match the current model contract.

    Current PhoBERTClassifier already places dropout inside ClassificationHead,
    so we must call the head directly on pooled features.
    """
    if hasattr(model, "classifier"):
        return model.classifier(pooled)
    if hasattr(model, "head"):
        return model.head(pooled)
    raise AttributeError("Text model must have either `classifier` or `head`.")


def export_split(
    model,
    loader,
    device: torch.device,
    output_path: Path,
    compact_tokens: int,
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
    }

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            token_type_ids = batch.get("token_type_ids")

            encoder_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask.to(device) if attention_mask is not None else None,
                "return_dict": True,
            }
            if token_type_ids is not None:
                encoder_kwargs["token_type_ids"] = token_type_ids.to(device)

            outputs = model.encoder(**encoder_kwargs)
            hidden_states = outputs.last_hidden_state
            attn_mask = encoder_kwargs["attention_mask"]
            pooled = model._pool(hidden_states, attention_mask=attn_mask)
            logits = run_classifier_head(model, pooled)
            probs = torch.softmax(logits, dim=-1)

            batch_ids = [str(x) for x in batch["ids"]]
            batch_labels = batch["labels"].tolist()
            batch_texts = [str(x) for x in batch.get("texts", [])]
            batch_raw_texts = [str(x) for x in batch.get("raw_texts", [])]
            batch_audio_paths = [str(x) for x in batch.get("audio_paths", [])]
            batch_group_ids = [str(x) for x in batch.get("group_ids", [])]

            if attn_mask is None:
                attn_mask = torch.ones(
                    hidden_states.shape[:2],
                    dtype=torch.long,
                    device=hidden_states.device,
                )

            for i in range(hidden_states.shape[0]):
                compact_tokens_i, compact_mask_i = compress_sequence_to_token_bank(
                    sequence=hidden_states[i],
                    mask=attn_mask[i],
                    target_len=compact_tokens,
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
    validate_expert_export_artifact(artifact, "text_export")

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
    config, path_patched = patch_text_dataset_paths_if_legacy(config, PROJECT_ROOT)
    config, model_patched = normalize_legacy_model_keys(config)

    if path_patched:
        print("[INFO] Patched legacy text dataset paths to current canonical VNEMOS CSVs.")
    if model_patched:
        print("[INFO] Patched legacy text model registry key to current codebase naming.")

    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)

    tokenizer_dir = run_dir / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = build_text_tokenizer(
            tokenizer_config=config["tokenizer"],
            model_config=config["model"],
        )

    train_loader, valid_loader, test_loader = build_text_dataloaders(
        config=config,
        tokenizer=tokenizer,
    )

    model = build_model(config)
    ckpt_path = find_best_checkpoint(run_dir)
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    notes = load_checkpoint_state_flexibly(model, ckpt)
    for note in notes:
        print(f"[INFO] Adapted legacy checkpoint key: {note}")
    model.to(device)

    export_split(model, train_loader, device, args.output_dir / "train.pt", args.compact_tokens)
    export_split(model, valid_loader, device, args.output_dir / "valid.pt", args.compact_tokens)
    export_split(model, test_loader, device, args.output_dir / "test.pt", args.compact_tokens)

    print(f"[OK] Exported text features to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())