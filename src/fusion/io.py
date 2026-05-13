from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.fusion.features import (
    apply_temperature_scaling,
    build_reliability_vector,
    logits_to_probs,
    load_calibration_state,
)


def load_torch_artifact(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Artifact not found: {file_path}")
    artifact = torch.load(file_path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise TypeError(f"Expected dict artifact at {file_path}, got {type(artifact)}")
    return artifact


def save_torch_artifact(artifact: dict[str, Any], path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, file_path)


def _index_by_sample_id(artifact: dict[str, Any]) -> dict[str, int]:
    sample_ids = artifact["sample_id"]
    return {str(sample_id): idx for idx, sample_id in enumerate(sample_ids)}


def _get_tensor_row(tensor: torch.Tensor, idx: int) -> torch.Tensor:
    return tensor[idx].detach().cpu()


def _validate_required_keys(
    artifact: dict[str, Any],
    required_keys: list[str],
    artifact_name: str,
) -> None:
    missing = [key for key in required_keys if key not in artifact]
    if missing:
        raise ValueError(f"{artifact_name} artifact missing required keys: {missing}")


def _validate_tensor_first_dim(
    tensor: torch.Tensor,
    expected_n: int,
    tensor_name: str,
    artifact_name: str,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{artifact_name}.{tensor_name} must be a torch.Tensor, got {type(tensor)}")
    if tensor.shape[0] != expected_n:
        raise ValueError(
            f"{artifact_name}.{tensor_name} first dim mismatch: "
            f"expected {expected_n}, got {tuple(tensor.shape)}"
        )


def _assert_finite(tensor: torch.Tensor, tensor_name: str, artifact_name: str) -> None:
    if not torch.isfinite(tensor).all():
        bad = (~torch.isfinite(tensor)).nonzero(as_tuple=False)
        raise ValueError(
            f"{artifact_name}.{tensor_name} contains NaN/Inf values. "
            f"First bad index: {bad[0].tolist() if bad.numel() > 0 else 'unknown'}"
        )


def _assert_binary_mask(mask: torch.Tensor, mask_name: str, artifact_name: str) -> None:
    unique_vals = torch.unique(mask)
    allowed = {0, 1}
    got = set(int(v) for v in unique_vals.tolist())
    if not got.issubset(allowed):
        raise ValueError(
            f"{artifact_name}.{mask_name} must be binary mask in {{0,1}}, got values={sorted(got)}"
        )


def sanitize_compact_token_bank(
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.nan_to_num(tokens.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mask = (mask > 0).long()

    if int(mask.sum().item()) <= 0:
        mask = mask.clone()
        tokens = tokens.clone()
        mask[0] = 1
        tokens[0].zero_()

    tokens = tokens * mask.unsqueeze(-1).float()
    return tokens.contiguous(), mask.contiguous()


def validate_expert_export_artifact(
    artifact: dict[str, Any],
    artifact_name: str,
) -> None:
    required_keys = [
        "sample_id",
        "label_id",
        "logits",
        "probs",
        "pooled_embedding",
        "compact_tokens",
        "compact_token_masks",
    ]
    _validate_required_keys(artifact, required_keys, artifact_name)

    sample_ids = artifact["sample_id"]
    expected_n = len(sample_ids)
    if expected_n <= 0:
        raise ValueError(f"{artifact_name} artifact has no samples.")

    label_id = artifact["label_id"]
    logits = artifact["logits"]
    probs = artifact["probs"]
    pooled_embedding = artifact["pooled_embedding"]
    compact_tokens = artifact["compact_tokens"]
    compact_token_masks = artifact["compact_token_masks"]

    _validate_tensor_first_dim(label_id, expected_n, "label_id", artifact_name)
    _validate_tensor_first_dim(logits, expected_n, "logits", artifact_name)
    _validate_tensor_first_dim(probs, expected_n, "probs", artifact_name)
    _validate_tensor_first_dim(pooled_embedding, expected_n, "pooled_embedding", artifact_name)
    _validate_tensor_first_dim(compact_tokens, expected_n, "compact_tokens", artifact_name)
    _validate_tensor_first_dim(compact_token_masks, expected_n, "compact_token_masks", artifact_name)

    if label_id.dim() != 1:
        raise ValueError(f"{artifact_name}.label_id must be 1D, got {tuple(label_id.shape)}")
    if logits.dim() != 2:
        raise ValueError(f"{artifact_name}.logits must be 2D, got {tuple(logits.shape)}")
    if probs.dim() != 2:
        raise ValueError(f"{artifact_name}.probs must be 2D, got {tuple(probs.shape)}")
    if pooled_embedding.dim() != 2:
        raise ValueError(
            f"{artifact_name}.pooled_embedding must be 2D, got {tuple(pooled_embedding.shape)}"
        )
    if compact_tokens.dim() != 3:
        raise ValueError(
            f"{artifact_name}.compact_tokens must be 3D [N,K,D], got {tuple(compact_tokens.shape)}"
        )
    if compact_token_masks.dim() != 2:
        raise ValueError(
            f"{artifact_name}.compact_token_masks must be 2D [N,K], got {tuple(compact_token_masks.shape)}"
        )
    if compact_tokens.shape[:2] != compact_token_masks.shape:
        raise ValueError(
            f"{artifact_name}.compact_tokens and compact_token_masks shape mismatch: "
            f"{tuple(compact_tokens.shape)} vs {tuple(compact_token_masks.shape)}"
        )

    _assert_finite(logits, "logits", artifact_name)
    _assert_finite(probs, "probs", artifact_name)
    _assert_finite(pooled_embedding, "pooled_embedding", artifact_name)
    _assert_finite(compact_tokens, "compact_tokens", artifact_name)
    _assert_binary_mask(compact_token_masks, "compact_token_masks", artifact_name)

    valid_counts = compact_token_masks.sum(dim=1)
    if (valid_counts <= 0).any():
        bad_rows = (valid_counts <= 0).nonzero(as_tuple=False).view(-1).tolist()
        examples = [str(sample_ids[i]) for i in bad_rows[:5]]
        raise ValueError(
            f"{artifact_name}.compact_token_masks has samples with zero valid tokens. "
            f"Examples: {examples}"
        )


def validate_fusion_ready_artifact(artifact: dict[str, Any]) -> None:
    required_keys = [
        "sample_id",
        "label_id",
        "text_logits_raw",
        "speech_logits_raw",
        "text_probs_cal",
        "speech_probs_cal",
        "text_embedding",
        "speech_embedding",
        "text_tokens",
        "speech_tokens",
        "text_token_masks",
        "speech_token_masks",
        "reliability",
    ]
    _validate_required_keys(artifact, required_keys, "fusion_ready")

    sample_ids = artifact["sample_id"]
    expected_n = len(sample_ids)
    if expected_n <= 0:
        raise ValueError("fusion_ready artifact has no samples.")

    tensor_keys = [
        "label_id",
        "text_logits_raw",
        "speech_logits_raw",
        "text_probs_cal",
        "speech_probs_cal",
        "text_embedding",
        "speech_embedding",
        "text_tokens",
        "speech_tokens",
        "text_token_masks",
        "speech_token_masks",
        "reliability",
    ]

    optional_tensor_keys = [
        "text_logits_cal",
        "speech_logits_cal",
    ]

    for key in tensor_keys:
        _validate_tensor_first_dim(artifact[key], expected_n, key, "fusion_ready")

    for key in optional_tensor_keys:
        if key in artifact:
            _validate_tensor_first_dim(artifact[key], expected_n, key, "fusion_ready")

    if artifact["label_id"].dim() != 1:
        raise ValueError("fusion_ready.label_id must be 1D")

    for key in [
        "text_logits_raw",
        "speech_logits_raw",
        "text_probs_cal",
        "speech_probs_cal",
        "text_embedding",
        "speech_embedding",
        "reliability",
    ]:
        if artifact[key].dim() != 2:
            raise ValueError(f"fusion_ready.{key} must be 2D, got {tuple(artifact[key].shape)}")

    for key in optional_tensor_keys:
        if key in artifact and artifact[key].dim() != 2:
            raise ValueError(f"fusion_ready.{key} must be 2D, got {tuple(artifact[key].shape)}")

    for key in ["text_tokens", "speech_tokens"]:
        if artifact[key].dim() != 3:
            raise ValueError(f"fusion_ready.{key} must be 3D, got {tuple(artifact[key].shape)}")

    for key in ["text_token_masks", "speech_token_masks"]:
        if artifact[key].dim() != 2:
            raise ValueError(f"fusion_ready.{key} must be 2D, got {tuple(artifact[key].shape)}")

    if artifact["text_tokens"].shape[:2] != artifact["text_token_masks"].shape:
        raise ValueError("fusion_ready.text_tokens/text_token_masks shape mismatch")
    if artifact["speech_tokens"].shape[:2] != artifact["speech_token_masks"].shape:
        raise ValueError("fusion_ready.speech_tokens/speech_token_masks shape mismatch")

    for key in [
        "text_logits_raw",
        "speech_logits_raw",
        "text_probs_cal",
        "speech_probs_cal",
        "text_embedding",
        "speech_embedding",
        "text_tokens",
        "speech_tokens",
        "reliability",
    ]:
        _assert_finite(artifact[key], key, "fusion_ready")

    for key in optional_tensor_keys:
        if key in artifact:
            _assert_finite(artifact[key], key, "fusion_ready")

    _assert_binary_mask(artifact["text_token_masks"], "text_token_masks", "fusion_ready")
    _assert_binary_mask(artifact["speech_token_masks"], "speech_token_masks", "fusion_ready")

    bad_text = (artifact["text_token_masks"].sum(dim=1) <= 0).nonzero(as_tuple=False).view(-1).tolist()
    bad_speech = (artifact["speech_token_masks"].sum(dim=1) <= 0).nonzero(as_tuple=False).view(-1).tolist()
    if bad_text:
        raise ValueError(
            f"fusion_ready.text_token_masks has zero-valid-token samples. "
            f"Examples: {[str(sample_ids[i]) for i in bad_text[:5]]}"
        )
    if bad_speech:
        raise ValueError(
            f"fusion_ready.speech_token_masks has zero-valid-token samples. "
            f"Examples: {[str(sample_ids[i]) for i in bad_speech[:5]]}"
        )


def prepare_joined_fusion_artifact(
    text_artifact: dict[str, Any],
    speech_artifact: dict[str, Any],
    reliability_config: dict[str, Any],
    text_calibration_state: dict[str, Any] | None = None,
    speech_calibration_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_expert_export_artifact(text_artifact, "text_export")
    validate_expert_export_artifact(speech_artifact, "speech_export")

    text_index = _index_by_sample_id(text_artifact)
    speech_index = _index_by_sample_id(speech_artifact)

    common_ids = [sample_id for sample_id in text_artifact["sample_id"] if str(sample_id) in speech_index]
    if not common_ids:
        raise ValueError("No common sample_id found between text and speech artifacts.")

    sample_ids: list[str] = []
    label_ids: list[int] = []

    text_logits_raw: list[torch.Tensor] = []
    speech_logits_raw: list[torch.Tensor] = []
    text_logits_cal: list[torch.Tensor] = []
    speech_logits_cal: list[torch.Tensor] = []
    text_probs_cal: list[torch.Tensor] = []
    speech_probs_cal: list[torch.Tensor] = []

    text_embeddings: list[torch.Tensor] = []
    speech_embeddings: list[torch.Tensor] = []

    text_tokens: list[torch.Tensor] = []
    speech_tokens: list[torch.Tensor] = []
    text_token_masks: list[torch.Tensor] = []
    speech_token_masks: list[torch.Tensor] = []

    metadata = {
        "text": [],
        "raw_text": [],
        "audio_path": [],
        "group_id": [],
        "duration": [],
    }

    for sample_id in common_ids:
        ti = text_index[str(sample_id)]
        si = speech_index[str(sample_id)]

        text_label = int(text_artifact["label_id"][ti])
        speech_label = int(speech_artifact["label_id"][si])
        if text_label != speech_label:
            raise ValueError(
                f"Label mismatch for sample_id={sample_id}: text={text_label}, speech={speech_label}"
            )

        z_t_raw = _get_tensor_row(text_artifact["logits"], ti).float()
        z_s_raw = _get_tensor_row(speech_artifact["logits"], si).float()

        z_t_cal = apply_temperature_scaling(z_t_raw.unsqueeze(0), text_calibration_state).squeeze(0)
        z_s_cal = apply_temperature_scaling(z_s_raw.unsqueeze(0), speech_calibration_state).squeeze(0)

        p_t_cal = logits_to_probs(z_t_cal.unsqueeze(0)).squeeze(0)
        p_s_cal = logits_to_probs(z_s_cal.unsqueeze(0)).squeeze(0)

        text_tok, text_mask = sanitize_compact_token_bank(
            _get_tensor_row(text_artifact["compact_tokens"], ti).float(),
            _get_tensor_row(text_artifact["compact_token_masks"], ti).long(),
        )
        speech_tok, speech_mask = sanitize_compact_token_bank(
            _get_tensor_row(speech_artifact["compact_tokens"], si).float(),
            _get_tensor_row(speech_artifact["compact_token_masks"], si).long(),
        )

        sample_ids.append(str(sample_id))
        label_ids.append(text_label)

        text_logits_raw.append(z_t_raw)
        speech_logits_raw.append(z_s_raw)
        text_logits_cal.append(z_t_cal)
        speech_logits_cal.append(z_s_cal)
        text_probs_cal.append(p_t_cal)
        speech_probs_cal.append(p_s_cal)

        text_embeddings.append(_get_tensor_row(text_artifact["pooled_embedding"], ti).float())
        speech_embeddings.append(_get_tensor_row(speech_artifact["pooled_embedding"], si).float())

        text_tokens.append(text_tok)
        speech_tokens.append(speech_tok)
        text_token_masks.append(text_mask)
        speech_token_masks.append(speech_mask)

        text_meta = text_artifact.get("metadata", {})
        speech_meta = speech_artifact.get("metadata", {})

        metadata["text"].append(str(text_meta.get("text", [""])[ti]) if "text" in text_meta else "")
        metadata["raw_text"].append(str(text_meta.get("raw_text", [""])[ti]) if "raw_text" in text_meta else "")
        metadata["audio_path"].append(
            str(speech_meta.get("audio_path", [""])[si]) if "audio_path" in speech_meta else ""
        )
        metadata["group_id"].append(
            str(speech_meta.get("group_id", [""])[si]) if "group_id" in speech_meta else ""
        )
        metadata["duration"].append(
            float(speech_meta.get("duration", [0.0])[si]) if "duration" in speech_meta else 0.0
        )

    text_probs_tensor = torch.stack(text_probs_cal, dim=0)
    speech_probs_tensor = torch.stack(speech_probs_cal, dim=0)

    reliability = build_reliability_vector(
        text_probs=text_probs_tensor,
        speech_probs=speech_probs_tensor,
        use_confidence=bool(reliability_config.get("use_confidence", True)),
        use_entropy=bool(reliability_config.get("use_entropy", True)),
        use_margin=bool(reliability_config.get("use_margin", True)),
        use_variance=bool(reliability_config.get("use_variance", False)),
    )

    artifact = {
        "sample_id": sample_ids,
        "label_id": torch.tensor(label_ids, dtype=torch.long),
        "text_logits_raw": torch.stack(text_logits_raw, dim=0),
        "speech_logits_raw": torch.stack(speech_logits_raw, dim=0),
        "text_logits_cal": torch.stack(text_logits_cal, dim=0),
        "speech_logits_cal": torch.stack(speech_logits_cal, dim=0),
        "text_probs_cal": text_probs_tensor,
        "speech_probs_cal": speech_probs_tensor,
        "text_embedding": torch.stack(text_embeddings, dim=0),
        "speech_embedding": torch.stack(speech_embeddings, dim=0),
        "text_tokens": torch.stack(text_tokens, dim=0),
        "speech_tokens": torch.stack(speech_tokens, dim=0),
        "text_token_masks": torch.stack(text_token_masks, dim=0),
        "speech_token_masks": torch.stack(speech_token_masks, dim=0),
        "reliability": reliability,
        "metadata": metadata,
    }
    validate_fusion_ready_artifact(artifact)
    return artifact


def load_expert_artifacts_and_prepare(
    text_artifact_path: str | Path,
    speech_artifact_path: str | Path,
    reliability_config: dict[str, Any],
    text_calibration_path: str | Path | None = None,
    speech_calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    text_artifact = load_torch_artifact(text_artifact_path)
    speech_artifact = load_torch_artifact(speech_artifact_path)

    text_calibration_state = load_calibration_state(text_calibration_path)
    speech_calibration_state = load_calibration_state(speech_calibration_path)

    return prepare_joined_fusion_artifact(
        text_artifact=text_artifact,
        speech_artifact=speech_artifact,
        reliability_config=reliability_config,
        text_calibration_state=text_calibration_state,
        speech_calibration_state=speech_calibration_state,
    )