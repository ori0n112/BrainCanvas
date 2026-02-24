from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LossConfig
from engine.metrics import retrieval_metrics
from models.neural_decoder import DecoderOutput
from utils import safe_soft_clip_loss


class Objective:
    def __init__(self, cfg: LossConfig) -> None:
        self.cfg = cfg
        self._mse = nn.MSELoss()
        self._ce = nn.CrossEntropyLoss()

    def __call__(
        self,
        *,
        voxel: torch.Tensor,
        clip_image_target: torch.Tensor,
        clip_text_target: torch.Tensor,
        output: DecoderOutput,
        adversarial_weight: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        metrics: dict[str, float] = {}
        adv_weight = self.cfg.adversarial_weight if adversarial_weight is None else float(adversarial_weight)

        image_pred = F.normalize(output.image_embedding.flatten(1), dim=-1)
        image_target = F.normalize(clip_image_target.flatten(1), dim=-1)
        text_pred = F.normalize(output.text_embedding.flatten(1), dim=-1)
        text_target = F.normalize(clip_text_target.flatten(1), dim=-1)

        image_clip_loss = safe_soft_clip_loss(image_pred, image_target)
        text_clip_loss = safe_soft_clip_loss(text_pred, text_target)
        image_mse = self._mse(image_pred, image_target)
        text_mse = self._mse(text_pred, text_target)

        image_loss = image_clip_loss + image_mse
        text_loss = text_clip_loss + text_mse

        total = self.cfg.image_weight * image_loss + self.cfg.text_weight * text_loss

        metrics["loss_clip_image"] = image_clip_loss.item()
        metrics["loss_clip_text"] = text_clip_loss.item()

        metrics["loss_mse_image"] = image_mse.item()
        metrics["loss_mse_text"] = text_mse.item()
        metrics["loss_image"] = image_loss.item()
        metrics["loss_text"] = text_loss.item()

        if output.voxel_reconstruction is not None:
            reconstruction = self._mse(voxel, output.voxel_reconstruction)
            total = total + reconstruction
            metrics["loss_reconstruction"] = reconstruction.item()

        if adv_weight > 0 and output.domain_logits is not None and output.domain_targets is not None:
            domain_loss = self._ce(output.domain_logits, output.domain_targets)
            total = total + adv_weight * domain_loss
            metrics["loss_adversarial"] = domain_loss.item()
            metrics["domain_accuracy"] = (
                (output.domain_logits.argmax(dim=-1) == output.domain_targets).float().mean().item()
            )

        metrics["adversarial_weight"] = adv_weight

        metrics["loss_total"] = total.item()

        image_retrieval = retrieval_metrics(output.image_embedding, clip_image_target)
        text_retrieval = retrieval_metrics(output.text_embedding, clip_text_target)
        metrics["cosine_image"] = image_retrieval["cosine_mean"]
        metrics["cosine_text"] = text_retrieval["cosine_mean"]
        metrics["top1_image_forward"] = image_retrieval["top1_forward"]
        metrics["top1_image_backward"] = image_retrieval["top1_backward"]
        metrics["top1_text_forward"] = text_retrieval["top1_forward"]
        metrics["top1_text_backward"] = text_retrieval["top1_backward"]

        return total, metrics
