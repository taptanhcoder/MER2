from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import load_yaml
from src.utils.paths import resolve_project_path


def find_best_checkpoint(run_dir: Path) -> Path:
    candidates = [
        run_dir / "best.pt",
        run_dir / "best.ckpt",
        run_dir / "checkpoint_best.pt",
        run_dir / "model_best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path

    recursive = sorted(list(run_dir.rglob("*best*.pt")) + list(run_dir.rglob("*best*.ckpt")))
    if recursive:
        return recursive[0]

    recursive_last = sorted(list(run_dir.rglob("*last*.pt")) + list(run_dir.rglob("*last*.ckpt")))
    if recursive_last:
        return recursive_last[0]

    raise FileNotFoundError(f"Could not find best checkpoint under: {run_dir}")


def load_run_or_explicit_config(
    run_dir: Path,
    explicit_config_path: str | Path | None,
    project_root: str | Path,
) -> dict[str, Any]:
    if explicit_config_path is not None:
        config = load_yaml(resolve_project_path(explicit_config_path, project_root))
    else:
        config_path = run_dir / "resolved_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Could not find resolved_config.yaml in run dir: {run_dir}. "
                "Either provide a run with full artifacts or pass --config explicitly."
            )
        config = load_yaml(config_path)

    config.setdefault("project", {})
    config["project"].setdefault("root", str(project_root))
    return config


def _path_exists(path_value: str | Path | None, project_root: str | Path) -> bool:
    if path_value is None:
        return False
    try:
        return resolve_project_path(path_value, project_root).exists()
    except Exception:
        return False


def normalize_legacy_model_keys(config: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    cfg = dict(config)
    model_cfg = dict(cfg.get("model", {}))

    legacy_to_current = {
        "phobert-base-v2": "phobert-base",
        "phobert-base-v1": "phobert-base",
        "hubert-base-ls960": "hubert-base",
        "hubert-base-ls960-ft": "hubert-base",
    }

    current_name = model_cfg.get("name")
    changed = False
    if current_name in legacy_to_current:
        model_cfg["name"] = legacy_to_current[str(current_name)]
        changed = True

    cfg["model"] = model_cfg
    return cfg, changed


def patch_text_dataset_paths_if_legacy(
    config: dict[str, Any],
    project_root: str | Path,
) -> tuple[dict[str, Any], bool]:
    cfg = dict(config)
    dataset_cfg = dict(cfg.get("dataset", {}))

    expected_paths = {
        "train_csv": "data/splits/vnemos_train.csv",
        "valid_csv": "data/splits/vnemos_valid.csv",
        "test_csv": "data/splits/vnemos_test.csv",
    }

    changed = False
    for key, fallback in expected_paths.items():
        current = dataset_cfg.get(key)
        if not _path_exists(current, project_root):
            fallback_resolved = resolve_project_path(fallback, project_root)
            if fallback_resolved.exists():
                dataset_cfg[key] = fallback
                changed = True

    dataset_cfg.setdefault("id_col", "sample_id")
    dataset_cfg.setdefault("text_col", "text")
    dataset_cfg.setdefault("raw_text_col", "transcript_final")
    dataset_cfg.setdefault("normalized_text_col", "normalized_text")
    dataset_cfg.setdefault("audio_path_col", "audio_path")
    dataset_cfg.setdefault("group_id_col", "group_id")
    dataset_cfg.setdefault("label_col", "label")
    dataset_cfg.setdefault("label_id_col", "label_id")
    dataset_cfg.setdefault("num_classes", 5)
    dataset_cfg.setdefault("use_normalized_text", False)

    cfg["dataset"] = dataset_cfg
    return cfg, changed


def patch_speech_dataset_paths_if_legacy(
    config: dict[str, Any],
    project_root: str | Path,
) -> tuple[dict[str, Any], bool]:
    cfg = dict(config)
    dataset_cfg = dict(cfg.get("dataset", {}))

    expected_paths = {
        "train_csv": "data/splits/vnemos_train.csv",
        "valid_csv": "data/splits/vnemos_valid.csv",
        "test_csv": "data/splits/vnemos_test.csv",
    }

    changed = False
    for key, fallback in expected_paths.items():
        current = dataset_cfg.get(key)
        if not _path_exists(current, project_root):
            fallback_resolved = resolve_project_path(fallback, project_root)
            if fallback_resolved.exists():
                dataset_cfg[key] = fallback
                changed = True

    dataset_cfg.setdefault("id_col", "sample_id")
    dataset_cfg.setdefault("audio_path_col", "audio_path")
    dataset_cfg.setdefault("duration_col", "duration")
    dataset_cfg.setdefault("group_id_col", "group_id")
    dataset_cfg.setdefault("text_col", "text")
    dataset_cfg.setdefault("raw_text_col", "transcript_final")
    dataset_cfg.setdefault("label_col", "label")
    dataset_cfg.setdefault("label_id_col", "label_id")
    dataset_cfg.setdefault("num_classes", 5)

    cfg["dataset"] = dataset_cfg
    return cfg, changed


def _find_first_linear_index_in_sequential(module: nn.Module) -> int | None:
    if not hasattr(module, "net"):
        return None
    net = getattr(module, "net")
    if not isinstance(net, nn.Sequential):
        return None

    for idx, submodule in enumerate(net):
        if isinstance(submodule, nn.Linear):
            return idx
    return None


def adapt_legacy_state_dict_for_model(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    adapted = dict(state_dict)
    notes: list[str] = []

    linear_index = None
    if hasattr(model, "classifier"):
        linear_index = _find_first_linear_index_in_sequential(model.classifier)

    legacy_weight = "head.classifier.weight"
    legacy_bias = "head.classifier.bias"
    if linear_index is not None:
        new_weight = f"classifier.net.{linear_index}.weight"
        new_bias = f"classifier.net.{linear_index}.bias"

        if legacy_weight in adapted and new_weight not in adapted:
            adapted[new_weight] = adapted.pop(legacy_weight)
            notes.append(f"{legacy_weight} -> {new_weight}")

        if legacy_bias in adapted and new_bias not in adapted:
            adapted[new_bias] = adapted.pop(legacy_bias)
            notes.append(f"{legacy_bias} -> {new_bias}")

    return adapted, notes


def load_checkpoint_state_flexibly(
    model: nn.Module,
    checkpoint: dict[str, Any] | dict[str, torch.Tensor],
) -> list[str]:
    if "model_state_dict" in checkpoint:
        raw_state_dict = checkpoint["model_state_dict"]
    else:
        raw_state_dict = checkpoint

    if not isinstance(raw_state_dict, dict):
        raise TypeError(f"Expected checkpoint state_dict to be dict, got {type(raw_state_dict)}")

    adapted_state_dict, notes = adapt_legacy_state_dict_for_model(model, raw_state_dict)

    incompatible = model.load_state_dict(adapted_state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint still incompatible after legacy adaptation.\n"
            f"Missing keys: {list(incompatible.missing_keys)}\n"
            f"Unexpected keys: {list(incompatible.unexpected_keys)}"
        )

    return notes


def compress_sequence_to_token_bank(
    sequence: torch.Tensor,
    mask: torch.Tensor | None,
    target_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compress a variable-length sequence [L, D] into a compact token bank [K, D].

    Returns:
        compact_tokens: [K, D]
        compact_mask:   [K]
    """
    if sequence.dim() != 2:
        raise ValueError(f"Expected sequence shape [L, D], got {tuple(sequence.shape)}")
    if target_len <= 0:
        raise ValueError("target_len must be positive")

    device = sequence.device
    dtype = sequence.dtype
    seq_len, hidden_dim = sequence.shape

    if mask is None:
        mask = torch.ones(seq_len, dtype=torch.long, device=device)
    else:
        mask = mask.long().to(device)

    valid_len = int(mask.sum().item())
    valid_len = max(valid_len, 1)
    valid_seq = sequence[:valid_len]  # [Lv, D]

    if valid_len > target_len:
        compact = F.adaptive_avg_pool1d(
            valid_seq.transpose(0, 1).unsqueeze(0),  # [1, D, Lv]
            output_size=target_len,
        ).squeeze(0).transpose(0, 1)  # [K, D]
        compact_mask = torch.ones(target_len, dtype=torch.long, device=device)
    else:
        compact = torch.zeros(target_len, hidden_dim, dtype=dtype, device=device)
        compact[:valid_len] = valid_seq
        compact_mask = torch.zeros(target_len, dtype=torch.long, device=device)
        compact_mask[:valid_len] = 1

    return compact, compact_mask