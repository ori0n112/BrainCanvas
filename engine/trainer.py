from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

import torch
from tqdm import tqdm

from config import AppConfig, Architecture, SchedulerType
from data.dataset import maybe_augment_images
from data.coco_prompts import COCOTextRepository
from engine.batching import Batch, BatchStream
from engine.checkpoint import CheckpointManager
from engine.losses import Objective
from utils import MetricTracker


class Trainer:
    def __init__(
        self,
        *,
        config: AppConfig,
        accelerator,
        model: torch.nn.Module,
        clip_extractor,
        batch_stream: BatchStream,
        prompt_repo: COCOTextRepository,
    ) -> None:
        self.config = config
        self.accelerator = accelerator
        self.model = model
        self.clip_extractor = clip_extractor
        self.batch_stream = batch_stream
        self.prompt_repo = prompt_repo
        self.objective = Objective(config.loss)

        self._adversarial_target_weight = self._resolve_adversarial_target_weight()

        self.checkpoints = CheckpointManager(config.run_dir, accelerator)
        self.start_epoch = 0
        self.best_score = -float("inf")
        self.best_epoch = -1

        self.optimizer = self._build_optimizer(model)
        self.scheduler = self._build_scheduler(
            optimizer=self.optimizer,
            epochs=config.optim.epochs,
            steps_per_epoch=batch_stream.train_steps,
        )

        if config.runtime.load_from is not None:
            self.checkpoints.load_weights(self.model, config.runtime.load_from)

        self.model, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.scheduler,
        )
        self.batch_stream = self.batch_stream.prepare(self.accelerator)

        if config.runtime.resume:
            self.start_epoch = self.checkpoints.resume_training_state()

        self._wandb_run = None
        self._init_wandb()

    def _build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        no_decay = ("bias", "norm", "Norm", "temperature")
        with_decay = []
        without_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            bucket = without_decay if any(token in name for token in no_decay) else with_decay
            bucket.append(param)

        return torch.optim.AdamW(
            [
                {"params": with_decay, "weight_decay": self.config.optim.weight_decay},
                {"params": without_decay, "weight_decay": 0.0},
            ],
            lr=self.config.optim.max_lr,
        )

    def _resolve_adversarial_target_weight(self) -> float:
        return float(self.config.loss.adversarial_weight)

    def _scheduled_adversarial_weight(self) -> float:
        if self._adversarial_target_weight <= 0:
            return 0.0
        return self._adversarial_target_weight

    def _build_scheduler(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        steps_per_epoch: int,
    ):
        total_steps = max(1, epochs * steps_per_epoch)
        if self.config.optim.scheduler is SchedulerType.LINEAR:
            return torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=total_steps)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.optim.max_lr,
            total_steps=total_steps,
            final_div_factor=100,
            pct_start=max(1 / max(1, epochs), 0.01),
        )

    def _init_wandb(self) -> None:
        if not self.config.runtime.use_wandb or not self.accelerator.is_main_process:
            return

        import wandb

        run_kwargs = {
            "project": self.config.runtime.wandb_project,
            "name": self.config.runtime.run_name,
            "config": self._flatten_config(),
        }
        if self.config.runtime.resume and self.config.runtime.resume_id:
            run_kwargs["id"] = self.config.runtime.resume_id
            run_kwargs["resume"] = "allow"

        self._wandb_run = wandb.init(**run_kwargs)

    def _flatten_config(self) -> dict[str, object]:
        flattened: dict[str, object] = {}
        for section_name in ("runtime", "data", "model", "loss", "optim", "inference"):
            section = getattr(self.config, section_name)
            for key, value in asdict(section).items():
                flattened[f"{section_name}.{key}"] = value
        return flattened

    @staticmethod
    def _move_batch_to_device(batch: Batch, device: torch.device) -> Batch:
        return Batch(
            voxel=batch.voxel.to(device, non_blocking=True).float(),
            image=batch.image.to(device, non_blocking=True).float(),
            coco_id=batch.coco_id,
            subject_id=batch.subject_id.to(device, non_blocking=True).long(),
        )

    def _step(
        self,
        *,
        batch: Batch,
        repeat_index: int,
        training: bool,
        adversarial_weight: float,
    ) -> dict[str, float]:
        prepared = batch.training_view(repeat_index) if training else batch.validation_view()
        prepared = self._move_batch_to_device(prepared, self.accelerator.device)

        if training and self.config.data.use_image_augmentation:
            prepared = Batch(
                voxel=prepared.voxel,
                image=maybe_augment_images(prepared.image, True),
                coco_id=prepared.coco_id,
                subject_id=prepared.subject_id,
            )

        captions = self.prompt_repo.prompts_for_batch(prepared.coco_id, repeat_index)

        with torch.no_grad():
            clip_image_target = self.clip_extractor.encode_images(prepared.image).float()
            clip_text_target = self.clip_extractor.encode_texts(captions).float()

        if training:
            self.optimizer.zero_grad(set_to_none=True)

        output = self.model(prepared.voxel, prepared.subject_id)
        loss, metrics = self.objective(
            voxel=prepared.voxel,
            clip_image_target=clip_image_target,
            clip_text_target=clip_text_target,
            output=output,
            adversarial_weight=adversarial_weight,
        )

        if training:
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.scheduler.step()

        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        return metrics

    def _run_epoch(self, *, epoch: int, training: bool) -> dict[str, float]:
        self.model.train(training)
        tracker = MetricTracker()

        iterator = self.batch_stream.train_batches() if training else self.batch_stream.val_batches()
        description = f"epoch={epoch} train" if training else f"epoch={epoch} val"
        progress = tqdm(iterator, disable=not self.accelerator.is_main_process, desc=description)

        for step, batch in enumerate(progress):
            repeat = step % 3
            adversarial_weight = self._scheduled_adversarial_weight()
            with torch.set_grad_enabled(training):
                metrics = self._step(
                    batch=batch,
                    repeat_index=repeat,
                    training=training,
                    adversarial_weight=adversarial_weight,
                )
            tracker.update(metrics)

            if self.accelerator.is_main_process:
                progress.set_postfix({"loss": tracker.average("loss_total"), "lr": tracker.average("lr")})

        return tracker.snapshot()

    def _log(self, payload: Mapping[str, float], *, step: int) -> None:
        if not self.config.runtime.use_wandb or self._wandb_run is None:
            return

        import wandb

        wandb.log(dict(payload), step=step)

    @staticmethod
    def _prefix_metrics(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
        return {f"{prefix}/{key}": value for key, value in metrics.items()}

    def fit(self) -> None:
        self.accelerator.print(
            f"Starting run={self.config.runtime.run_name} at epoch={self.start_epoch}/{self.config.optim.epochs}"
        )
        self.accelerator.print(
            "Adversarial setup: "
            f"target_weight={self._adversarial_target_weight:.4f}"
        )

        for epoch in range(self.start_epoch, self.config.optim.epochs):
            train_metrics = self._run_epoch(epoch=epoch, training=True)
            logs = self._prefix_metrics("train", train_metrics)

            if epoch % self.config.optim.eval_every == 0:
                val_metrics = self._run_epoch(epoch=epoch, training=False)
                logs.update(self._prefix_metrics("val", val_metrics))

                score = float(val_metrics.get("cosine_image", 0.0) + val_metrics.get("cosine_text", 0.0))
                if score > self.best_score:
                    self.best_score = score
                    self.best_epoch = epoch
                    self.checkpoints.save(tag="best", epoch=epoch, model=self.model)

            logs["epoch"] = float(epoch)
            self._log(logs, step=epoch)

            if epoch % self.config.optim.checkpoint_every == 0 or epoch == self.config.optim.epochs - 1:
                self.checkpoints.save(tag="last", epoch=epoch, model=self.model)

            self.accelerator.wait_for_everyone()

        self.accelerator.print(
            f"Training complete. best_epoch={self.best_epoch}, best_score={self.best_score:.4f}"
        )
