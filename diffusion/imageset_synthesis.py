from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from diffusion.attention_control import (
    AttentionStoreClassPrompts,
    StoredAttnClassPromptsProcessor,
    aggregate_attention_maps,
    register_attention_controller,
)
from models import CategoryAwareLocalizationModule


@dataclass(slots=True)
class PromptRecord:
    filename: str
    caption: str
    class_prompt: str = ""
    token_groups: list[int | list[int]] | None = None
    class_ids: list[int] | None = None


def load_prompt_records(json_path: Path) -> list[PromptRecord]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    items = payload["prompts"] if isinstance(payload, dict) and "prompts" in payload else payload
    records: list[PromptRecord] = []
    for entry in items:
        records.append(
            PromptRecord(
                filename=str(entry.get("filename", f"sample_{len(records):06d}.png")),
                caption=str(entry.get("caption", "")),
                class_prompt=str(entry.get("class_prompt", "")),
                token_groups=entry.get("token_groups") or entry.get("indices"),
                class_ids=entry.get("class_ids") or entry.get("labels"),
            )
        )
    return records


def _as_pil_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    array = np.asarray(value)
    if array.dtype != np.uint8:
        array = np.clip(array, 0.0, 1.0)
        array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)


def _compose_prompt(caption: str, class_prompt: str, delimiter: str = "; ") -> str:
    class_prompt = class_prompt.strip()
    if not class_prompt:
        return caption
    return f"{caption}{delimiter}{class_prompt}"


def _build_generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(int(seed))


class ImageSetSynthesizer:
    """Class-prompt guided synthetic image generation and mask refinement."""

    def __init__(
        self,
        *,
        sd_model_path: str,
        sd_inpainting_model_path: str | None,
        device: torch.device,
        seed: int,
        self_attention_resolution: int,
        cross_attention_resolution: int,
        threshold: float,
        uncertainty_threshold: float | None,
        start_step: int,
        end_step: int,
        inpaint_enabled: bool,
    ) -> None:
        self.sd_model_path = sd_model_path
        self.sd_inpainting_model_path = sd_inpainting_model_path
        self.device = device
        self.generator = _build_generator(seed)
        self.self_attention_resolution = self_attention_resolution
        self.cross_attention_resolution = cross_attention_resolution
        self.threshold = threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.start_step = start_step
        self.end_step = end_step
        self.inpaint_enabled = inpaint_enabled

        self.localization = CategoryAwareLocalizationModule(power=4)

        self.text2image_pipe = None
        self.inpaint_pipe = None
        self.controller: AttentionStoreClassPrompts | None = None

    def initialize(self) -> None:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.text2image_pipe = self._load_text2image_pipeline(dtype=dtype)
        self.inpaint_pipe = self._load_inpainting_pipeline(dtype=dtype)
        self.controller = AttentionStoreClassPrompts(start_step=self.start_step, end_step=self.end_step)
        register_attention_controller(self.text2image_pipe, self.controller, StoredAttnClassPromptsProcessor)

    def _load_text2image_pipeline(self, *, dtype: torch.dtype):
        from diffusers import StableDiffusionPipeline

        pipe = StableDiffusionPipeline.from_pretrained(self.sd_model_path, torch_dtype=dtype).to(self.device)
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        return pipe

    def _load_inpainting_pipeline(self, *, dtype: torch.dtype):
        if not self.inpaint_enabled or not self.sd_inpainting_model_path:
            return None

        try:
            from diffusers import AutoPipelineForInpainting

            pipe = AutoPipelineForInpainting.from_pretrained(self.sd_inpainting_model_path, torch_dtype=dtype)
        except Exception:
            from diffusers import StableDiffusionInpaintPipeline

            pipe = StableDiffusionInpaintPipeline.from_pretrained(self.sd_inpainting_model_path, torch_dtype=dtype)

        pipe = pipe.to(self.device)
        if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
            try:
                pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        return pipe

    def _collect_attention(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.controller is None:
            raise RuntimeError("Attention controller is not initialized")

        self_attention = aggregate_attention_maps(
            self.controller,
            resolution=self.self_attention_resolution,
            is_cross=False,
        ).float()
        cross_attention = aggregate_attention_maps(
            self.controller,
            resolution=self.cross_attention_resolution,
            is_cross=True,
        ).float()
        cross_attention = F.interpolate(
            cross_attention.permute(0, 3, 1, 2),
            size=(self.self_attention_resolution, self.self_attention_resolution),
            mode="bicubic",
        )
        return self_attention, cross_attention

    @staticmethod
    def _inpaint_mask_from_semantic_mask(semantic_mask: np.ndarray) -> Image.Image:
        binary = ((semantic_mask > 0) & (semantic_mask != 255)).astype(np.uint8) * 255
        mask = Image.fromarray(binary, mode="L")
        return ImageOps.invert(mask)

    def synthesize(
        self,
        *,
        records: list[PromptRecord],
        output_dir: Path,
        batch_size: int,
        num_inference_steps: int,
        negative_prompt: str,
        start_index: int,
        end_index: int | None,
    ) -> None:
        if self.text2image_pipe is None or self.controller is None:
            self.initialize()
        assert self.text2image_pipe is not None
        assert self.controller is not None
        text2image_pipe = self.text2image_pipe
        controller = self.controller

        image_dir = output_dir / "image"
        mask_dir = output_dir / "mask"
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        stop = len(records) if end_index is None else min(len(records), end_index)
        start = max(0, start_index)

        for offset in range(start, stop, batch_size):
            batch = records[offset : offset + batch_size]
            prompts = [_compose_prompt(item.caption, item.class_prompt) for item in batch]

            output = text2image_pipe(
                prompts,
                negative_prompt=[negative_prompt] * len(batch),
                num_inference_steps=num_inference_steps,
                generator=self.generator,
                output_type="numpy",
            )

            images = output.images
            try:
                self_attention, cross_attention = self._collect_attention()
            except RuntimeError:
                self_attention, cross_attention = None, None

            for index, record in enumerate(batch):
                base_name = Path(record.filename).stem
                image = _as_pil_image(images[index])
                image.save(image_dir / f"{base_name}.jpg")

                has_localization_targets = (
                    self_attention is not None
                    and cross_attention is not None
                    and record.token_groups is not None
                    and len(record.token_groups) > 0
                    and record.class_ids is not None
                    and len(record.class_ids) > 1
                )
                if not has_localization_targets:
                    continue

                assert self_attention is not None
                assert cross_attention is not None
                assert record.token_groups is not None
                assert record.class_ids is not None

                width, height = image.size
                localized = self.localization.localize(
                    self_attention=self_attention[index],
                    cross_attention=cross_attention[index],
                    token_groups=record.token_groups,
                )
                semantic_mask = self.localization.to_semantic_mask(
                    localized_maps=localized,
                    class_ids=[int(v) for v in record.class_ids],
                    out_height=height,
                    out_width=width,
                    threshold=self.threshold,
                    uncertainty_threshold=self.uncertainty_threshold,
                )
                Image.fromarray(semantic_mask).save(mask_dir / f"{base_name}_semantic.png")

                if self.inpaint_pipe is None:
                    continue

                inpaint_mask = self._inpaint_mask_from_semantic_mask(semantic_mask)
                refined = self.inpaint_pipe(
                    prompt=_compose_prompt(record.caption, record.class_prompt),
                    image=image,
                    mask_image=inpaint_mask,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    generator=self.generator,
                ).images[0]
                refined.save(mask_dir / f"{base_name}.png")

            controller.reset()
