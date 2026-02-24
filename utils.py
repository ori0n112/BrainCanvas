from __future__ import annotations

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


def seed_everything(seed: int, *, deterministic_cudnn: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = deterministic_cudnn


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def safe_soft_clip_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.005) -> torch.Tensor:
    eps = 1e-10
    target_logits = (target @ target.T) / temperature + eps
    pred_logits = (pred @ target.T) / temperature + eps

    forward = -(pred_logits.log_softmax(dim=-1) * target_logits.softmax(dim=-1)).sum(dim=-1).mean()
    backward = -(pred_logits.T.log_softmax(dim=-1) * target_logits.softmax(dim=-1)).sum(dim=-1).mean()
    return 0.5 * (forward + backward)


def cosine_batch_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_norm = torch.linalg.norm(a, dim=1, keepdim=True)
    b_norm = torch.linalg.norm(b, dim=1, keepdim=True)
    return (a @ b.T) / (a_norm @ b_norm.T + 1e-8)


def topk_accuracy(similarity: torch.Tensor, labels: torch.Tensor, k: int = 1) -> float:
    if k <= 0:
        return 0.0
    k = min(k, similarity.shape[1])
    topk = torch.topk(similarity, k=k, dim=1).indices
    matches = (topk == labels.unsqueeze(1)).any(dim=1)
    return matches.float().mean().item()


@dataclass(slots=True)
class MetricTracker:
    _sum: dict[str, float]
    _count: dict[str, int]

    def __init__(self) -> None:
        self._sum = defaultdict(float)
        self._count = defaultdict(int)

    def update(self, metrics: Mapping[str, float], *, n: int = 1) -> None:
        for key, value in metrics.items():
            self._sum[key] += float(value) * n
            self._count[key] += n

    def average(self, key: str) -> float:
        if self._count[key] == 0:
            return 0.0
        return self._sum[key] / self._count[key]

    def snapshot(self) -> dict[str, float]:
        return {k: self.average(k) for k in self._sum}
