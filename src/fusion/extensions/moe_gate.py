from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def load_torch_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Fusion artifact not found: {artifact_path}")
    return torch.load(artifact_path, map_location="cpu")


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def _entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -(probs * np.log(np.clip(probs, eps, 1.0))).sum(axis=1, keepdims=True)


def _confidence(probs: np.ndarray) -> np.ndarray:
    return probs.max(axis=1, keepdims=True)


def _margin(probs: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probs, axis=1)
    if sorted_probs.shape[1] == 1:
        return sorted_probs[:, -1:].copy()
    return (sorted_probs[:, -1] - sorted_probs[:, -2]).reshape(-1, 1)


def _pred_id(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(axis=1).astype("int64")


def _one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    output = np.zeros((indices.shape[0], num_classes), dtype="float32")
    output[np.arange(indices.shape[0]), indices.astype("int64")] = 1.0
    return output


def get_labels(artifact: dict[str, Any]) -> np.ndarray:
    if "label_id" not in artifact:
        raise KeyError("Fusion artifact missing `label_id`.")
    return _to_numpy(artifact["label_id"]).astype("int64")


def get_sample_ids(artifact: dict[str, Any]) -> list[str]:
    if "sample_id" not in artifact:
        return [str(i) for i in range(len(get_labels(artifact)))]
    return [str(x) for x in artifact["sample_id"]]


def get_expert_logits(
    artifact: dict[str, Any],
    prefix: str,
    prefer_calibrated: bool = True,
) -> np.ndarray:
    calibrated_key = f"{prefix}_logits_cal"
    raw_key = f"{prefix}_logits_raw"

    if prefer_calibrated and calibrated_key in artifact:
        return _to_numpy(artifact[calibrated_key]).astype("float32")
    if raw_key in artifact:
        return _to_numpy(artifact[raw_key]).astype("float32")
    if calibrated_key in artifact:
        return _to_numpy(artifact[calibrated_key]).astype("float32")

    raise KeyError(f"Missing `{calibrated_key}` or `{raw_key}` in artifact.")


def get_expert_probs(
    artifact: dict[str, Any],
    prefix: str,
    prefer_calibrated: bool = True,
) -> np.ndarray:
    calibrated_key = f"{prefix}_probs_cal"
    raw_key = f"{prefix}_probs"

    if prefer_calibrated and calibrated_key in artifact:
        return _to_numpy(artifact[calibrated_key]).astype("float32")
    if raw_key in artifact:
        return _to_numpy(artifact[raw_key]).astype("float32")
    if calibrated_key in artifact:
        return _to_numpy(artifact[calibrated_key]).astype("float32")

    logits = get_expert_logits(
        artifact=artifact,
        prefix=prefix,
        prefer_calibrated=prefer_calibrated,
    )
    return _softmax(logits).astype("float32")


def build_moe_features(
    artifact: dict[str, Any],
    use_logits: bool = True,
    use_probs: bool = True,
    use_reliability: bool = True,
    use_derived: bool = True,
    use_agreement: bool = True,
    use_pred_onehot: bool = False,
    prefer_calibrated: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels = get_labels(artifact)

    text_logits = get_expert_logits(
        artifact,
        "text",
        prefer_calibrated=prefer_calibrated,
    )
    speech_logits = get_expert_logits(
        artifact,
        "speech",
        prefer_calibrated=prefer_calibrated,
    )
    text_probs = get_expert_probs(
        artifact,
        "text",
        prefer_calibrated=prefer_calibrated,
    )
    speech_probs = get_expert_probs(
        artifact,
        "speech",
        prefer_calibrated=prefer_calibrated,
    )

    if text_probs.shape != speech_probs.shape:
        raise ValueError(
            f"Mismatched expert probability shapes: "
            f"text={text_probs.shape}, speech={speech_probs.shape}"
        )

    num_classes = int(text_probs.shape[1])
    feature_blocks: list[np.ndarray] = []
    feature_names: list[str] = []

    if use_probs:
        feature_blocks.extend([text_probs, speech_probs])
        feature_names.extend([f"text_prob_{idx}" for idx in range(num_classes)])
        feature_names.extend([f"speech_prob_{idx}" for idx in range(num_classes)])

    if use_logits:
        feature_blocks.extend([text_logits, speech_logits])
        feature_names.extend([f"text_logit_{idx}" for idx in range(num_classes)])
        feature_names.extend([f"speech_logit_{idx}" for idx in range(num_classes)])

    text_pred = _pred_id(text_probs)
    speech_pred = _pred_id(speech_probs)

    if use_pred_onehot:
        text_onehot = _one_hot(text_pred, num_classes)
        speech_onehot = _one_hot(speech_pred, num_classes)
        feature_blocks.extend([text_onehot, speech_onehot])
        feature_names.extend([f"text_pred_onehot_{idx}" for idx in range(num_classes)])
        feature_names.extend([f"speech_pred_onehot_{idx}" for idx in range(num_classes)])

    if use_reliability and "reliability" in artifact:
        reliability = _to_numpy(artifact["reliability"]).astype("float32")
        feature_blocks.append(reliability)
        feature_names.extend(
            [f"reliability_{idx}" for idx in range(reliability.shape[1])]
        )

    if use_derived:
        derived_blocks = [
            _confidence(text_probs),
            _confidence(speech_probs),
            _entropy(text_probs),
            _entropy(speech_probs),
            _margin(text_probs),
            _margin(speech_probs),
        ]
        feature_blocks.extend(derived_blocks)
        feature_names.extend(
            [
                "text_confidence",
                "speech_confidence",
                "text_entropy",
                "speech_entropy",
                "text_margin",
                "speech_margin",
            ]
        )

    if use_agreement:
        agreement = (text_pred == speech_pred).astype("float32").reshape(-1, 1)
        feature_blocks.append(agreement)
        feature_names.append("text_speech_agreement")

    if not feature_blocks:
        raise ValueError("No MoE features selected.")

    features = np.concatenate(feature_blocks, axis=1).astype("float32")
    return features, labels, feature_names


@dataclass
class StandardizerState:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype("float32")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.astype("float32").tolist(),
            "std": self.std.astype("float32").tolist(),
        }


def fit_standardizer(x: np.ndarray) -> StandardizerState:
    mean = x.mean(axis=0, keepdims=True).astype("float32")
    std = x.std(axis=0, keepdims=True).astype("float32")
    std = np.where(std < 1e-6, 1.0, std).astype("float32")
    return StandardizerState(mean=mean, std=std)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


@dataclass
class MoEGateConfig:
    input_dim: int
    num_classes: int = 5
    gate_type: str = "classwise"
    hidden_dim: int = 64
    dropout: float = 0.10
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-3
    epochs: int = 250
    patience: int = 30
    batch_size: int = 16
    seed: int = 42
    gate_entropy_lambda: float = 0.0
    gate_balance_lambda: float = 0.0


class TwoExpertMoEGate(nn.Module):
    """
    Two-expert MoE gate over frozen text/speech logits.

    gate_type="classwise":
        alpha shape = [batch, num_classes, 2]
        each class has its own text/speech mixture weights.

    gate_type="scalar":
        alpha shape = [batch, num_classes, 2], but weights are shared
        across classes for each sample.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        gate_type: str,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        gate_type = str(gate_type).strip().lower()
        if gate_type not in {"classwise", "scalar"}:
            raise ValueError(
                f"Unsupported gate_type={gate_type!r}. "
                "Supported values: classwise, scalar."
            )

        self.num_classes = int(num_classes)
        self.gate_type = gate_type
        output_dim = self.num_classes * 2 if gate_type == "classwise" else 2

        self.gate = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(output_dim)),
        )

    def forward(
        self,
        x: torch.Tensor,
        text_logits: torch.Tensor,
        speech_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(x.shape[0])
        gate_logits = self.gate(x)

        if self.gate_type == "classwise":
            alpha = torch.softmax(
                gate_logits.view(batch_size, self.num_classes, 2),
                dim=-1,
            )
        else:
            scalar_alpha = torch.softmax(
                gate_logits.view(batch_size, 1, 2),
                dim=-1,
            )
            alpha = scalar_alpha.expand(batch_size, self.num_classes, 2)

        fused_logits = (
            alpha[..., 0] * text_logits
            + alpha[..., 1] * speech_logits
        )
        return fused_logits, alpha


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_tensor(
    value: np.ndarray,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(value, dtype=dtype, device=device)


def _macro_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()
    return float(f1_score(y_true, preds, average="macro", zero_division=0))


def _gate_entropy(alpha: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    return -(alpha * torch.log(torch.clamp(alpha, min=eps))).sum(dim=-1).mean()


def _gate_balance_loss(alpha: torch.Tensor) -> torch.Tensor:
    mean_alpha = alpha.mean(dim=(0, 1))
    target = torch.full_like(mean_alpha, 0.5)
    return F.mse_loss(mean_alpha, target)


def _minibatch_indices(num_rows: int, batch_size: int, device: torch.device) -> list[torch.Tensor]:
    indices = torch.randperm(num_rows, device=device)
    batches = []
    for start in range(0, num_rows, int(batch_size)):
        batches.append(indices[start : start + int(batch_size)])
    return batches


def train_moe_gate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    text_logits_train: np.ndarray,
    speech_logits_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    text_logits_valid: np.ndarray,
    speech_logits_valid: np.ndarray,
    config: MoEGateConfig,
    device: str = "cuda",
) -> tuple[TwoExpertMoEGate, dict[str, Any]]:
    _set_seed(int(config.seed))

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    model = TwoExpertMoEGate(
        input_dim=int(config.input_dim),
        num_classes=int(config.num_classes),
        gate_type=str(config.gate_type),
        hidden_dim=int(config.hidden_dim),
        dropout=float(config.dropout),
    ).to(torch_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.lr),
        weight_decay=float(config.weight_decay),
    )

    x_tr = _to_tensor(x_train, torch.float32, torch_device)
    y_tr = _to_tensor(y_train, torch.long, torch_device)
    zt_tr = _to_tensor(text_logits_train, torch.float32, torch_device)
    zs_tr = _to_tensor(speech_logits_train, torch.float32, torch_device)

    x_va = _to_tensor(x_valid, torch.float32, torch_device)
    y_va = _to_tensor(y_valid, torch.long, torch_device)
    zt_va = _to_tensor(text_logits_valid, torch.float32, torch_device)
    zs_va = _to_tensor(speech_logits_valid, torch.float32, torch_device)

    best_state: dict[str, torch.Tensor] | None = None
    best_valid_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    history = []

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        epoch_losses = []

        for idx in _minibatch_indices(
            num_rows=x_tr.shape[0],
            batch_size=int(config.batch_size),
            device=torch_device,
        ):
            optimizer.zero_grad(set_to_none=True)

            logits, alpha = model(x_tr[idx], zt_tr[idx], zs_tr[idx])
            loss = F.cross_entropy(logits, y_tr[idx])

            if float(config.gate_entropy_lambda) > 0.0:
                loss = loss + float(config.gate_entropy_lambda) * _gate_entropy(alpha)

            if float(config.gate_balance_lambda) > 0.0:
                loss = loss + float(config.gate_balance_lambda) * _gate_balance_loss(alpha)

            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            valid_logits, valid_alpha = model(x_va, zt_va, zs_va)
            valid_loss = float(F.cross_entropy(valid_logits, y_va).detach().cpu().item())
            valid_f1 = _macro_f1_from_logits(valid_logits, y_va)

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "valid_loss": valid_loss,
                "valid_macro_f1": valid_f1,
                "valid_alpha_text_mean": float(valid_alpha[..., 0].mean().detach().cpu().item()),
                "valid_alpha_speech_mean": float(valid_alpha[..., 1].mean().detach().cpu().item()),
            }
        )

        if valid_f1 > best_valid_f1:
            best_valid_f1 = valid_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= int(config.patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    report = {
        "best_valid_macro_f1": float(best_valid_f1),
        "best_epoch": int(best_epoch),
        "history": history,
    }
    return model, report


def predict_moe_gate(
    model: TwoExpertMoEGate,
    x: np.ndarray,
    text_logits: np.ndarray,
    speech_logits: np.ndarray,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    model.to(torch_device)
    model.eval()

    with torch.no_grad():
        xt = _to_tensor(x, torch.float32, torch_device)
        zt = _to_tensor(text_logits, torch.float32, torch_device)
        zs = _to_tensor(speech_logits, torch.float32, torch_device)

        logits, alpha = model(xt, zt, zs)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

    return (
        probs.detach().cpu().numpy(),
        preds.detach().cpu().numpy(),
        alpha.detach().cpu().numpy(),
    )


def alpha_summary(alpha: np.ndarray) -> dict[str, Any]:
    text_alpha = alpha[..., 0]
    speech_alpha = alpha[..., 1]

    summary: dict[str, Any] = {
        "alpha_text_mean": float(text_alpha.mean()),
        "alpha_text_std": float(text_alpha.std()),
        "alpha_speech_mean": float(speech_alpha.mean()),
        "alpha_speech_std": float(speech_alpha.std()),
    }

    for class_idx in range(alpha.shape[1]):
        summary[f"alpha_text_class_{class_idx}"] = float(text_alpha[:, class_idx].mean())
        summary[f"alpha_speech_class_{class_idx}"] = float(speech_alpha[:, class_idx].mean())

    return summary


def save_predictions_csv(
    output_path: str | Path,
    sample_ids: list[str],
    labels: np.ndarray,
    probs: np.ndarray,
    alpha: np.ndarray,
    method_name: str,
    seed: int,
) -> None:
    preds = probs.argmax(axis=1)

    rows = []
    for idx, sample_id in enumerate(sample_ids):
        rows.append(
            {
                "sample_id": str(sample_id),
                "label_id": int(labels[idx]),
                "pred_id": int(preds[idx]),
                "confidence": float(probs[idx].max()),
                "probs": json.dumps(probs[idx].tolist(), ensure_ascii=False),
                "alpha_text_mean": float(alpha[idx, :, 0].mean()),
                "alpha_speech_mean": float(alpha[idx, :, 1].mean()),
                "method": method_name,
                "seed": int(seed),
            }
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_json(data: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")