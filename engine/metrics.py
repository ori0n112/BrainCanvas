from __future__ import annotations

import torch
import torch.nn.functional as F

from utils import cosine_batch_similarity, topk_accuracy


def normalize_pair(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_norm = F.normalize(pred.flatten(1), dim=-1)
    target_norm = F.normalize(target.flatten(1), dim=-1)
    return pred_norm, target_norm


def retrieval_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_norm, target_norm = normalize_pair(pred, target)
    similarity_forward = cosine_batch_similarity(pred_norm, target_norm)
    similarity_backward = cosine_batch_similarity(target_norm, pred_norm)
    labels = torch.arange(pred_norm.shape[0], device=pred_norm.device)

    return {
        "cosine_mean": torch.mean(torch.sum(pred_norm * target_norm, dim=-1)).item(),
        "top1_forward": topk_accuracy(similarity_forward, labels, k=1),
        "top1_backward": topk_accuracy(similarity_backward, labels, k=1),
    }
