from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def merge_caption_with_categories(
    caption: str,
    categories: Iterable[str] | None,
    *,
    delimiter: str = "; ",
    max_categories: int = 8,
) -> str:
    if categories is None:
        return caption
    tokens = [str(token).strip() for token in categories if str(token).strip()]
    if len(tokens) == 0:
        return caption
    tokens = list(dict.fromkeys(tokens))
    if max_categories > 0:
        tokens = tokens[:max_categories]
    return f"{caption}{delimiter}{', '.join(tokens)}"


def _vector_to_map(vector: torch.Tensor) -> torch.Tensor:
    side = int(math.sqrt(vector.shape[-1]))
    if side * side != vector.shape[-1]:
        raise ValueError(f"Cannot reshape length {vector.shape[-1]} into square map")
    return vector.view(vector.shape[0], 1, side, side)


class AttentionSpatializer(nn.Module):
    """Converts self/cross attention tensors to a spatial map."""

    @staticmethod
    def _to_map(attn: torch.Tensor) -> torch.Tensor:
        if attn.dim() == 4:
            # (B, H, Q, K) -> (B, K)
            return _vector_to_map(attn.mean(dim=1).mean(dim=1))
        if attn.dim() == 3 and attn.shape[-1] == attn.shape[-2]:
            # (B, Q, K) -> (B, K)
            return _vector_to_map(attn.mean(dim=1))
        if attn.dim() == 3:
            # (B, H, W)
            return attn.unsqueeze(1)
        if attn.dim() == 2:
            # (H, W)
            return attn.unsqueeze(0).unsqueeze(0)
        raise ValueError(f"Unsupported attention tensor shape: {tuple(attn.shape)}")

    def forward(self, attention_maps: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        if isinstance(attention_maps, torch.Tensor):
            return self._to_map(attention_maps)
        if len(attention_maps) == 0:
            raise ValueError("attention_maps cannot be empty")
        maps = [self._to_map(attn) for attn in attention_maps]
        return torch.stack(maps, dim=0).mean(dim=0)


class SelfCrossFusion(nn.Module):
    """Fuses self-attention and cross-attention maps into a position prior."""

    def __init__(self, channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.refine_self = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gate_cross = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.project = nn.Conv2d(channels * 2, out_channels, kernel_size=1)

    def forward(self, self_map: torch.Tensor, cross_map: torch.Tensor) -> torch.Tensor:
        if self_map.dim() == 3:
            self_map = self_map.unsqueeze(1)
        if cross_map.dim() == 3:
            cross_map = cross_map.unsqueeze(1)

        refined = self_map + torch.tanh(self.refine_self(self_map)) * torch.sigmoid(self.gate_cross(cross_map))
        return self.project(torch.cat([refined, cross_map], dim=1))


class StructuralBiasInjector(nn.Module):
    """Injects spatial structural priors into latent features."""

    def __init__(self, feature_channels: int, prior_channels: int = 1, strength: float = 1.0) -> None:
        super().__init__()
        self.strength = float(strength)
        self.projection = nn.Sequential(
            nn.Conv2d(prior_channels, feature_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, kernel_size=1),
        )

    def forward(self, latent_features: torch.Tensor, prior_map: torch.Tensor) -> torch.Tensor:
        if prior_map.dim() == 3:
            prior_map = prior_map.unsqueeze(1)
        prior_map = F.interpolate(prior_map, size=latent_features.shape[-2:], mode="bilinear", align_corners=False)
        bias = self.projection(prior_map)
        return latent_features + self.strength * bias


class CategoryAwareLocalizationModule:
    """Builds class-wise masks from self/cross attention maps.

    This module mirrors the key mechanism used by the paper implementation:
    self-attention provides affinity propagation while cross-attention seeds
    class-token responses.
    """

    def __init__(self, *, power: int = 4) -> None:
        self.power = int(power)

    @staticmethod
    def _normalize_map(attn_map: torch.Tensor) -> torch.Tensor:
        attn_map = attn_map - attn_map.min()
        denom = attn_map.max() + 1e-6
        return attn_map / denom

    @staticmethod
    def _as_token_group(value: int | list[int]) -> list[int]:
        if isinstance(value, list):
            return [int(v) + 1 for v in value]
        return [int(value) + 1]

    def localize(
        self,
        *,
        self_attention: torch.Tensor,
        cross_attention: torch.Tensor,
        token_groups: list[int | list[int]],
    ) -> torch.Tensor:
        """Computes class-localization heatmaps.

        Args:
            self_attention: `(H, W, HW)` or `(H, W, W)`-flattenable map.
            cross_attention: `(T, H, W)` token attention map.
            token_groups: token indices per class.
        """

        spatial_size = int(self_attention.shape[0])
        affinity = self_attention.reshape(spatial_size**2, spatial_size**2)
        affinity = torch.matrix_power(affinity, self.power)

        localized = []
        for group in token_groups:
            tokens = self._as_token_group(group)
            seed = cross_attention[tokens].mean(dim=0)
            score = (affinity @ seed.reshape(spatial_size**2, 1)).reshape(spatial_size, spatial_size)
            localized.append(self._normalize_map(score))

        return torch.stack(localized, dim=0)

    def to_semantic_mask(
        self,
        *,
        localized_maps: torch.Tensor,
        class_ids: list[int],
        out_height: int,
        out_width: int,
        threshold: float,
        uncertainty_threshold: float | None,
    ) -> np.ndarray:
        maps = F.interpolate(localized_maps.unsqueeze(0), (out_height, out_width), mode="bicubic")[0].cpu().numpy()
        maps_max = maps.max(axis=0)

        class_lookup = np.asarray(class_ids, dtype=np.uint8)
        mask_index = np.zeros((out_height, out_width), dtype=np.uint8)
        valid = maps_max >= threshold
        mask_index[valid] = (maps.argmax(axis=0) + 1)[valid]
        semantic_mask = class_lookup[mask_index]

        if uncertainty_threshold is not None:
            uncertain = (maps_max < threshold) & (maps_max >= uncertainty_threshold)
            semantic_mask[uncertain] = 255
        return semantic_mask
