from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Architecture, ModelConfig


@dataclass(slots=True)
class DecoderOutput:
    image_embedding: torch.Tensor
    text_embedding: torch.Tensor
    voxel_reconstruction: torch.Tensor | None = None
    cycle_loss: torch.Tensor | None = None
    domain_logits: torch.Tensor | None = None
    domain_targets: torch.Tensor | None = None


class BottleneckAdapter(nn.Module):
    def __init__(self, features: int, bottleneck: int = 128) -> None:
        super().__init__()
        self.down = nn.Linear(features, bottleneck)
        self.up = nn.Linear(bottleneck, features)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(x)))


class ResidualStack(nn.Module):
    def __init__(self, hidden_size: int, depth: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = x + block(x)
        return x


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainClassifier(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SingleDecoder(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        image_out_dim: int,
        text_out_dim: int,
        hidden_size: int,
        depth: int,
        subjects: Sequence[int],
    ) -> None:
        super().__init__()
        if len(subjects) == 0:
            raise ValueError("At least one subject is required")

        self.subjects = [int(s) for s in subjects]
        self.default_subject = self.subjects[0]
        self.hidden_size = hidden_size

        self.subject_encoders = nn.ModuleDict(
            {
                str(subject): nn.Sequential(
                    BottleneckAdapter(in_dim),
                    nn.Linear(in_dim, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.GELU(),
                    nn.Dropout(0.5),
                )
                for subject in self.subjects
            }
        )

        self.shared_backbone = ResidualStack(hidden_size, depth)
        self.image_head = nn.Linear(hidden_size, image_out_dim)
        self.text_head = nn.Linear(hidden_size, text_out_dim)

    def _subject_ids_or_default(self, batch_size: int, subject_ids: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if subject_ids is None:
            return torch.full((batch_size,), self.default_subject, dtype=torch.long, device=device)
        return subject_ids.long().view(-1)

    def _encode_subjectwise(self, voxels: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        features = torch.empty((voxels.shape[0], self.hidden_size), device=voxels.device, dtype=voxels.dtype)

        unique_subjects = torch.unique(subject_ids)
        for subject_id in unique_subjects.tolist():
            key = str(int(subject_id))
            if key not in self.subject_encoders:
                key = str(self.default_subject)
            mask = subject_ids == int(subject_id)
            features[mask] = self.subject_encoders[key](voxels[mask])
        return features

    def forward(self, voxels: torch.Tensor, subject_ids: torch.Tensor | None = None) -> DecoderOutput:
        subject_ids = self._subject_ids_or_default(voxels.shape[0], subject_ids, voxels.device)
        features = self._encode_subjectwise(voxels, subject_ids)
        shared = self.shared_backbone(features)

        return DecoderOutput(
            image_embedding=self.image_head(shared).reshape(len(shared), -1),
            text_embedding=self.text_head(shared).reshape(len(shared), -1),
        )


class FusionDecoder(SingleDecoder):
    def __init__(
        self,
        *,
        in_dim: int,
        image_out_dim: int,
        text_out_dim: int,
        hidden_size: int,
        depth: int,
        subjects: Sequence[int],
        adapting: bool,
    ) -> None:
        if len(subjects) < 2:
            raise ValueError("FusionDecoder requires at least two subjects")
        super().__init__(
            in_dim=in_dim,
            image_out_dim=image_out_dim,
            text_out_dim=text_out_dim,
            hidden_size=hidden_size,
            depth=depth,
            subjects=subjects,
        )
        self.adapting = adapting
        self.reconstructors = nn.ModuleDict(
            {
                str(subject): nn.Sequential(
                    nn.Linear(hidden_size, in_dim),
                    nn.LayerNorm(in_dim),
                    nn.GELU(),
                    BottleneckAdapter(in_dim),
                )
                for subject in self.subjects
            }
        )

    def _cycle_consistency(self, embeddings: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor | None:
        unique_subjects = torch.unique(subject_ids)
        if unique_subjects.numel() < 2:
            return None

        if self.adapting:
            source_subject = int(unique_subjects[0].item())
            target_subject = int(unique_subjects[-1].item())
        else:
            pair = random.sample([int(v.item()) for v in unique_subjects], k=2)
            source_subject, target_subject = pair[0], pair[1]

        source_mask = subject_ids == source_subject
        if not torch.any(source_mask):
            return None

        source_embedding = embeddings[source_mask]
        pseudo_voxel = self.reconstructors[str(target_subject)](source_embedding)
        pseudo_embedding = self.subject_encoders[str(target_subject)](pseudo_voxel)
        return F.mse_loss(source_embedding, pseudo_embedding)

    def forward(self, voxels: torch.Tensor, subject_ids: torch.Tensor | None = None) -> DecoderOutput:
        subject_ids = self._subject_ids_or_default(voxels.shape[0], subject_ids, voxels.device)
        embeddings = self._encode_subjectwise(voxels, subject_ids)

        recon = torch.empty_like(voxels)
        for subject_id in torch.unique(subject_ids).tolist():
            mask = subject_ids == int(subject_id)
            recon[mask] = self.reconstructors[str(int(subject_id))](embeddings[mask])

        cycle_loss = self._cycle_consistency(embeddings, subject_ids)
        shared = self.shared_backbone(embeddings)

        return DecoderOutput(
            image_embedding=self.image_head(shared).reshape(len(shared), -1),
            text_embedding=self.text_head(shared).reshape(len(shared), -1),
            voxel_reconstruction=recon,
            cycle_loss=cycle_loss,
        )


class BrainCanvasDecoder(FusionDecoder):
    def __init__(
        self,
        *,
        in_dim: int,
        image_out_dim: int,
        text_out_dim: int,
        hidden_size: int,
        depth: int,
        subjects: Sequence[int],
        adapting: bool,
        domain_hidden_size: int,
        domain_dropout: float,
        grl_lambda: float,
    ) -> None:
        super().__init__(
            in_dim=in_dim,
            image_out_dim=image_out_dim,
            text_out_dim=text_out_dim,
            hidden_size=hidden_size,
            depth=depth,
            subjects=subjects,
            adapting=adapting,
        )
        self.subject_to_domain = {subject: idx for idx, subject in enumerate(self.subjects)}
        self.grl = GradientReversal(grl_lambda)
        self.domain_head = DomainClassifier(
            in_dim=hidden_size,
            out_dim=len(self.subjects),
            hidden_dim=domain_hidden_size,
            dropout=domain_dropout,
        )

    def _domain_targets(self, subject_ids: torch.Tensor) -> torch.Tensor:
        mapped = torch.empty_like(subject_ids)
        for subject_id, domain in self.subject_to_domain.items():
            mapped[subject_ids == subject_id] = domain
        return mapped

    def forward(self, voxels: torch.Tensor, subject_ids: torch.Tensor | None = None) -> DecoderOutput:
        subject_ids = self._subject_ids_or_default(voxels.shape[0], subject_ids, voxels.device)
        embeddings = self._encode_subjectwise(voxels, subject_ids)

        recon = torch.empty_like(voxels)
        for subject_id in torch.unique(subject_ids).tolist():
            mask = subject_ids == int(subject_id)
            recon[mask] = self.reconstructors[str(int(subject_id))](embeddings[mask])

        shared = self.shared_backbone(embeddings)
        cycle_loss = self._cycle_consistency(embeddings, subject_ids)

        domain_targets = self._domain_targets(subject_ids)
        domain_logits = self.domain_head(self.grl(embeddings))

        return DecoderOutput(
            image_embedding=self.image_head(shared).reshape(len(shared), -1),
            text_embedding=self.text_head(shared).reshape(len(shared), -1),
            voxel_reconstruction=recon,
            cycle_loss=cycle_loss,
            domain_logits=domain_logits,
            domain_targets=domain_targets,
        )


def build_decoder(
    config: ModelConfig,
    *,
    subjects: Sequence[int],
    in_dim: int,
    image_out_dim: int,
    text_out_dim: int,
    adapting: bool,
) -> nn.Module:
    subjects = [int(s) for s in subjects]

    if len(subjects) == 1 or config.architecture is Architecture.SINGLE:
        return SingleDecoder(
            in_dim=in_dim,
            image_out_dim=image_out_dim,
            text_out_dim=text_out_dim,
            hidden_size=config.hidden_size,
            depth=config.depth,
            subjects=subjects,
        )

    if config.architecture is Architecture.BRAINCANVAS:
        return BrainCanvasDecoder(
            in_dim=in_dim,
            image_out_dim=image_out_dim,
            text_out_dim=text_out_dim,
            hidden_size=config.hidden_size,
            depth=config.depth,
            subjects=subjects,
            adapting=adapting,
            domain_hidden_size=config.domain_hidden_size,
            domain_dropout=config.domain_dropout,
            grl_lambda=config.grl_lambda,
        )

    return FusionDecoder(
        in_dim=in_dim,
        image_out_dim=image_out_dim,
        text_out_dim=text_out_dim,
        hidden_size=config.hidden_size,
        depth=config.depth,
        subjects=subjects,
        adapting=adapting,
    )
