from __future__ import annotations

from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.common.checkpoint import CheckpointManager
from src.common.interfaces import ClassifierOutput, EpochResult, PredictionBundle
from src.common.logging import build_logger
from src.common.losses import build_loss
from src.common.optimizers import build_optimizer
from src.common.schedulers import build_scheduler
from src.evaluation.metrics import compute_classification_metrics


class BaseTrainer(ABC):
    """Reusable training loop for text, speech, and fusion-style classifiers."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        train_loader: DataLoader | None = None,
        valid_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        class_names: list[str] | None = None,
        run_dir: str | Path = "outputs/runs/debug",
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        runtime_cfg = config.get("runtime", {})
        train_cfg = config.get("train", {})
        loss_cfg = config.get("loss", {"name": "cross_entropy"})
        optimizer_cfg = config.get("optimizer")
        scheduler_cfg = config.get("scheduler")
        logging_cfg = config.get("logging", {})
        checkpoint_cfg = config.get("checkpoint", {})
        early_stopping_cfg = config.get("early_stopping", {})

        requested_device = runtime_cfg.get("device", "cpu")
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        self.mixed_precision = str(runtime_cfg.get("mixed_precision", "none"))
        self.max_epochs = int(train_cfg.get("epochs", 1))
        self.grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0))
        self.grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
        self.primary_metric = str(config.get("evaluation", {}).get("primary_metric", "macro_f1"))
        self.log_every_n_steps = max(1, int(logging_cfg.get("log_every_n_steps", 20)))

        self.logger = build_logger(
            name=self.__class__.__name__,
            log_dir=self.run_dir,
            level=str(logging_cfg.get("level", "INFO")),
        )

        self.criterion = build_loss(loss_cfg)
        self.criterion.to(self.device)
        self.model.to(self.device)

        self.optimizer = None
        if optimizer_cfg is not None:
            self.optimizer = build_optimizer(optimizer_cfg, self.model.parameters())

        self.scheduler = None
        if scheduler_cfg is not None and self.optimizer is not None:
            scheduler_cfg = dict(scheduler_cfg)
            if scheduler_cfg.get("name") == "linear_warmup" and "total_steps" not in scheduler_cfg:
                if self.train_loader is None:
                    raise ValueError("linear_warmup scheduler requires train_loader or explicit total_steps")
                steps_per_epoch = ceil(max(1, len(self.train_loader)) / self.grad_accum_steps)
                scheduler_cfg["total_steps"] = self.max_epochs * steps_per_epoch
            self.scheduler = build_scheduler(scheduler_cfg, self.optimizer)

        try:
            self.scaler = torch.amp.GradScaler(
                "cuda",
                enabled=self.mixed_precision == "fp16" and self.device.type == "cuda",
            )
        except AttributeError:
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=self.mixed_precision == "fp16" and self.device.type == "cuda"
            )

        self.checkpoint_manager = CheckpointManager(
            run_dir=self.run_dir,
            monitor=checkpoint_cfg.get("monitor", self.primary_metric),
            mode=checkpoint_cfg.get("mode", "max"),
        )

        self.early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
        self.early_stopping_patience = int(early_stopping_cfg.get("patience", 5))
        self.early_stopping_min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
        self.early_stopping_monitor = str(early_stopping_cfg.get("monitor", self.primary_metric))
        self.early_stopping_mode = str(early_stopping_cfg.get("mode", "max"))
        self._epochs_without_improvement = 0
        self._best_early_stop_value: float | None = None

        self.logger.info(
            "device=%s mixed_precision=%s grad_accum_steps=%s",
            self.device,
            self.mixed_precision,
            self.grad_accum_steps,
        )

    @abstractmethod
    def prepare_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def forward_step(self, batch: dict[str, Any]) -> ClassifierOutput:
        raise NotImplementedError

    def compute_loss(self, output: ClassifierOutput, batch: dict[str, Any]) -> torch.Tensor:
        if output.loss is not None:
            return output.loss
        return self.criterion(output.logits, batch["labels"])

    def _is_early_stop_improved(self, value: float) -> bool:
        if self._best_early_stop_value is None:
            return True
        if self.early_stopping_mode == "min":
            return value < (self._best_early_stop_value - self.early_stopping_min_delta)
        return value > (self._best_early_stop_value + self.early_stopping_min_delta)

    def _run_loader(self, loader: DataLoader, split: str, training: bool) -> EpochResult:
        if training and self.optimizer is None:
            raise RuntimeError("Optimizer is required for training")

        self.model.train(training)
        total_loss = 0.0
        num_batches = 0

        all_ids: list[str] = []
        all_labels: list[int] = []
        all_preds: list[int] = []
        all_probs: list[list[float]] = []
        all_logits: list[list[float]] = []
        metadata_accumulator: dict[str, list[Any]] = {}

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(loader, start=1):
            prepared = self.prepare_batch(batch)
            labels = prepared["labels"]

            use_autocast = self.mixed_precision in {"fp16", "bf16"} and self.device.type == "cuda"
            autocast_dtype = torch.float16 if self.mixed_precision == "fp16" else torch.bfloat16

            with torch.autocast(
                device_type=self.device.type,
                dtype=autocast_dtype,
                enabled=use_autocast,
            ):
                output = self.forward_step(prepared)
                loss = self.compute_loss(output, prepared)

            loss_for_step = loss / self.grad_accum_steps if training else loss

            if training:
                if self.scaler.is_enabled():
                    self.scaler.scale(loss_for_step).backward()
                else:
                    loss_for_step.backward()

                should_step = (batch_idx % self.grad_accum_steps == 0) or (batch_idx == len(loader))
                if should_step:
                    if self.grad_clip_norm > 0:
                        if self.scaler.is_enabled():
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)

                    if self.scheduler is not None:
                        self.scheduler.step()

            probs = output.probs.detach().cpu()
            logits = output.logits.detach().cpu()
            preds = output.preds.detach().cpu()
            labels_cpu = labels.detach().cpu()

            sample_ids = prepared.get("ids")
            if sample_ids is None:
                sample_ids = [f"{split}_{num_batches}_{idx}" for idx in range(len(labels_cpu))]

            all_ids.extend([str(x) for x in sample_ids])
            all_labels.extend(labels_cpu.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())
            all_logits.extend(logits.tolist())

            prediction_metadata = prepared.get("prediction_metadata", {})
            if isinstance(prediction_metadata, dict):
                for key, values in prediction_metadata.items():
                    metadata_accumulator.setdefault(key, [])
                    metadata_accumulator[key].extend(list(values))

            total_loss += float(loss.detach().cpu().item())
            num_batches += 1

            if training and (batch_idx % self.log_every_n_steps == 0):
                self.logger.info(
                    "split=%s step=%s/%s loss=%.4f",
                    split,
                    batch_idx,
                    len(loader),
                    float(loss.detach().cpu().item()),
                )

        avg_loss = total_loss / max(1, num_batches)
        metrics = compute_classification_metrics(all_labels, all_preds, class_names=self.class_names)
        metrics["loss"] = avg_loss

        predictions = PredictionBundle(
            ids=all_ids,
            labels=all_labels,
            preds=all_preds,
            probs=all_probs,
            logits=all_logits,
            metadata=metadata_accumulator,
        )
        return EpochResult(split=split, loss=avg_loss, metrics=metrics, predictions=predictions)

    def train_epoch(self, loader: DataLoader) -> EpochResult:
        return self._run_loader(loader, split="train", training=True)

    def evaluate(self, loader: DataLoader, split: str = "valid") -> EpochResult:
        with torch.no_grad():
            return self._run_loader(loader, split=split, training=False)

    def fit(self) -> dict[str, Any]:
        if self.train_loader is None:
            raise RuntimeError("train_loader is required for fit()")

        history: list[dict[str, Any]] = []

        for epoch in range(1, self.max_epochs + 1):
            if hasattr(self.criterion, "set_epoch"):
                self.criterion.set_epoch(epoch)

            train_result = self.train_epoch(self.train_loader)
            log_payload: dict[str, Any] = {"epoch": epoch, "train": train_result.metrics}

            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "config": self.config,
            }
            self.checkpoint_manager.save_last(state)

            if self.valid_loader is not None:
                valid_result = self.evaluate(self.valid_loader, split="valid")
                log_payload["valid"] = valid_result.metrics

                monitored_value = float(valid_result.metrics[self.primary_metric])
                improved = self.checkpoint_manager.save_best_if_improved(state, value=monitored_value)

                self.logger.info(
                    "epoch=%s train_%s=%.4f valid_%s=%.4f improved=%s",
                    epoch,
                    self.primary_metric,
                    float(train_result.metrics[self.primary_metric]),
                    self.primary_metric,
                    monitored_value,
                    improved,
                )

                if self.early_stopping_enabled:
                    early_stop_value = float(valid_result.metrics[self.early_stopping_monitor])
                    if self._is_early_stop_improved(early_stop_value):
                        self._best_early_stop_value = early_stop_value
                        self._epochs_without_improvement = 0
                    else:
                        self._epochs_without_improvement += 1

                    if self._epochs_without_improvement >= self.early_stopping_patience:
                        self.logger.info(
                            "early_stopping_triggered epoch=%s monitor=%s best=%.4f patience=%s",
                            epoch,
                            self.early_stopping_monitor,
                            float(self._best_early_stop_value),
                            self.early_stopping_patience,
                        )
                        history.append(log_payload)
                        break
            else:
                self.logger.info(
                    "epoch=%s train_%s=%.4f",
                    epoch,
                    self.primary_metric,
                    float(train_result.metrics[self.primary_metric]),
                )

            history.append(log_payload)

        return {"history": history, "best_value": self.checkpoint_manager.best_value}