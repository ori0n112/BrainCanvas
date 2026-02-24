from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models

try:
    import clip as openai_clip
except Exception:
    openai_clip = None

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None


_SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".pt"}
_GROUND_TRUTH_DIRS = {"gt", "target", "targets", "ground_truth", "groundtruth"}
_RECON_DIRS = {"recon", "recons", "reconstruction", "reconstructions", "pred", "preds", "prediction"}

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)


@dataclass(slots=True)
class ImagePair:
    sample_id: str
    gt_path: Path
    recon_path: Path


def _strip_suffix_tag(stem: str, suffix_tag: str) -> str | None:
    if not stem.endswith(suffix_tag):
        return None
    value = stem[: -len(suffix_tag)]
    return value if value else None


def discover_image_pairs(results_path: Path) -> list[ImagePair]:
    gt_files: dict[str, Path] = {}
    recon_files: dict[str, Path] = {}

    for file_path in sorted(results_path.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in _SUPPORTED_SUFFIXES:
            continue

        stem = file_path.stem
        parent = file_path.parent.name.lower()

        gt_id = _strip_suffix_tag(stem, "_img")
        recon_id = _strip_suffix_tag(stem, "_recon")

        if gt_id is not None:
            gt_files.setdefault(gt_id, file_path)
            continue
        if recon_id is not None:
            recon_files.setdefault(recon_id, file_path)
            continue
        if parent in _GROUND_TRUTH_DIRS:
            gt_files.setdefault(stem, file_path)
            continue
        if parent in _RECON_DIRS:
            recon_files.setdefault(stem, file_path)

    shared_ids = sorted(set(gt_files.keys()) & set(recon_files.keys()))
    if len(shared_ids) == 0:
        raise RuntimeError(
            "No paired files found. Expected `<id>_img.*` and `<id>_recon.*`, "
            "or mirrored IDs under `gt/` and `recon/` directories."
        )

    return [ImagePair(sample_id=item, gt_path=gt_files[item], recon_path=recon_files[item]) for item in shared_ids]


def _to_3ch_unit_image(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.float().detach().cpu()

    if tensor.dim() == 4:
        if tensor.shape[0] == 0:
            raise ValueError("Image tensor cannot have an empty batch dimension")
        tensor = tensor[0]

    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)

    if tensor.dim() != 3:
        raise ValueError(f"Unsupported image tensor shape: {tuple(tensor.shape)}")

    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] > 3:
        tensor = tensor[:3]

    min_value = float(tensor.min().item())
    max_value = float(tensor.max().item())

    if min_value >= -1.0 and max_value <= 1.0 and min_value < 0.0:
        tensor = (tensor + 1.0) * 0.5
    elif max_value > 1.5:
        tensor = tensor / 255.0

    return tensor.clamp(0.0, 1.0)


def _read_image(path: Path) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            for key in ("image", "img", "recon", "prediction", "tensor", "data"):
                if key in payload:
                    payload = payload[key]
                    break
        tensor = torch.as_tensor(payload)
        return _to_3ch_unit_image(tensor)

    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.flatten().double()
    y = b.flatten().double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt((x * x).sum() * (y * y).sum()) + 1e-12
    value = (x * y).sum() / denominator
    return float(value.item())


def _global_ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    values: list[float] = []

    for channel in range(min(a.shape[0], b.shape[0])):
        x = a[channel].flatten().double()
        y = b[channel].flatten().double()
        mu_x = x.mean()
        mu_y = y.mean()
        sigma_x = ((x - mu_x) ** 2).mean()
        sigma_y = ((y - mu_y) ** 2).mean()
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        values.append(float((numerator / (denominator + 1e-12)).item()))

    return float(np.mean(values)) if values else 0.0


def _ssim_score(a: torch.Tensor, b: torch.Tensor) -> float:
    if skimage_ssim is None:
        return _global_ssim(a, b)

    a_np = a.permute(1, 2, 0).numpy()
    b_np = b.permute(1, 2, 0).numpy()
    return float(skimage_ssim(a_np, b_np, channel_axis=2, data_range=1.0))


def _pixel_metrics(pairs: list[ImagePair]) -> dict[str, float]:
    pixcorr_values: list[float] = []
    ssim_values: list[float] = []

    for pair in pairs:
        gt = _read_image(pair.gt_path)
        recon = _read_image(pair.recon_path)
        if recon.shape[-2:] != gt.shape[-2:]:
            recon = F.interpolate(
                recon.unsqueeze(0),
                size=gt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        pixcorr_values.append(_pearson_corr(gt, recon))
        ssim_values.append(_ssim_score(gt, recon))

    return {
        "PixCorr": float(np.mean(pixcorr_values)),
        "SSIM": float(np.mean(ssim_values)),
    }


def _build_preprocess(size: int, mean: torch.Tensor, std: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    mean = mean.view(3, 1, 1)
    std = std.view(3, 1, 1)

    def _apply(image: torch.Tensor) -> torch.Tensor:
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return (resized - mean) / std

    return _apply


def _default_encoder(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    return model(batch)


def _extract_features(
    pairs: list[ImagePair],
    *,
    select_path: Callable[[ImagePair], Path],
    model: torch.nn.Module,
    encoder: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    preprocess: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []

    with torch.no_grad():
        for offset in range(0, len(pairs), batch_size):
            chunk = pairs[offset : offset + batch_size]
            batch = torch.stack([preprocess(_read_image(select_path(pair))) for pair in chunk], dim=0)
            batch = batch.to(device, non_blocking=True)
            feature = encoder(model, batch)
            if not torch.is_tensor(feature):
                raise TypeError("Feature encoder must return a torch.Tensor")
            if feature.dim() > 2:
                feature = F.adaptive_avg_pool2d(feature, output_size=1).flatten(1)
            else:
                feature = feature.flatten(1)
            outputs.append(feature.detach().cpu())

    return torch.cat(outputs, dim=0)


def _top1_identification_percent(target: torch.Tensor, recon: torch.Tensor) -> float:
    target = F.normalize(target.float(), dim=-1)
    recon = F.normalize(recon.float(), dim=-1)
    similarity = recon @ target.T
    labels = torch.arange(similarity.shape[0])
    predictions = similarity.argmax(dim=1)
    return float((predictions == labels).float().mean().item() * 100.0)


def _paired_cosine_distance(target: torch.Tensor, recon: torch.Tensor) -> float:
    target = F.normalize(target.float(), dim=-1)
    recon = F.normalize(recon.float(), dim=-1)
    distance = 1.0 - torch.sum(recon * target, dim=-1)
    return float(distance.mean().item())


def _build_inception(device: torch.device):
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)
    return model, _default_encoder, _build_preprocess(299, _IMAGENET_MEAN, _IMAGENET_STD)


def _build_efficientnet(device: torch.device):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.classifier = nn.Identity()
    model.eval().to(device)
    return model, _default_encoder, _build_preprocess(224, _IMAGENET_MEAN, _IMAGENET_STD)


def _build_alexnet_layer(device: torch.device, stop_index: int):
    weights = models.AlexNet_Weights.IMAGENET1K_V1
    base = models.alexnet(weights=weights)
    model = nn.Sequential(base.features[: stop_index + 1], nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
    model.eval().to(device)
    return model, _default_encoder, _build_preprocess(224, _IMAGENET_MEAN, _IMAGENET_STD)


def _build_swav(device: torch.device):
    try:
        model = torch.hub.load("facebookresearch/swav:main", "resnet50")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load SwAV ResNet-50 via torch.hub. "
            "Ensure network access or pre-cache `facebookresearch/swav` models."
        ) from exc

    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    model.eval().to(device)
    return model, _default_encoder, _build_preprocess(224, _IMAGENET_MEAN, _IMAGENET_STD)


def _build_clip(device: torch.device):
    if openai_clip is None:
        raise RuntimeError("The `clip` package is required for CLIP metric evaluation")
    model, _ = openai_clip.load("ViT-B/32", device=device)
    model.eval()

    def _encoder(module: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
        return module.encode_image(batch)

    return model, _encoder, _build_preprocess(224, _CLIP_MEAN, _CLIP_STD)


def _evaluate_feature_metric(
    pairs: list[ImagePair],
    *,
    builder: Callable[[torch.device], tuple[torch.nn.Module, Callable[[torch.nn.Module, torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]],
    reduce_fn: Callable[[torch.Tensor, torch.Tensor], float],
    batch_size: int,
    device: torch.device,
) -> float:
    model, encoder, preprocess = builder(device)
    try:
        target = _extract_features(
            pairs,
            select_path=lambda pair: pair.gt_path,
            model=model,
            encoder=encoder,
            preprocess=preprocess,
            batch_size=batch_size,
            device=device,
        )
        recon = _extract_features(
            pairs,
            select_path=lambda pair: pair.recon_path,
            model=model,
            encoder=encoder,
            preprocess=preprocess,
            batch_size=batch_size,
            device=device,
        )
        return reduce_fn(target, recon)
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def evaluate_image_metrics(
    results_path: Path,
    *,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> tuple[dict[str, float], int]:
    pairs = discover_image_pairs(results_path)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    eval_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metrics: dict[str, float] = {}
    metrics.update(_pixel_metrics(pairs))
    metrics["Incep"] = _evaluate_feature_metric(
        pairs,
        builder=_build_inception,
        reduce_fn=_top1_identification_percent,
        batch_size=batch_size,
        device=eval_device,
    )
    metrics["CLIP"] = _evaluate_feature_metric(
        pairs,
        builder=_build_clip,
        reduce_fn=_top1_identification_percent,
        batch_size=batch_size,
        device=eval_device,
    )
    metrics["EffNet-B"] = _evaluate_feature_metric(
        pairs,
        builder=_build_efficientnet,
        reduce_fn=_paired_cosine_distance,
        batch_size=batch_size,
        device=eval_device,
    )
    metrics["SwAV"] = _evaluate_feature_metric(
        pairs,
        builder=_build_swav,
        reduce_fn=_paired_cosine_distance,
        batch_size=batch_size,
        device=eval_device,
    )
    metrics["Alex(2)"] = _evaluate_feature_metric(
        pairs,
        builder=lambda dev: _build_alexnet_layer(dev, stop_index=5),
        reduce_fn=_top1_identification_percent,
        batch_size=batch_size,
        device=eval_device,
    )
    metrics["Alex(5)"] = _evaluate_feature_metric(
        pairs,
        builder=lambda dev: _build_alexnet_layer(dev, stop_index=12),
        reduce_fn=_top1_identification_percent,
        batch_size=batch_size,
        device=eval_device,
    )

    ordered = {
        "Incep": metrics["Incep"],
        "CLIP": metrics["CLIP"],
        "EffNet-B": metrics["EffNet-B"],
        "SwAV": metrics["SwAV"],
        "PixCorr": metrics["PixCorr"],
        "SSIM": metrics["SSIM"],
        "Alex(2)": metrics["Alex(2)"],
        "Alex(5)": metrics["Alex(5)"],
    }
    return ordered, len(pairs)
