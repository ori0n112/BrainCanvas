from __future__ import annotations

from typing import Sequence

import clip
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms


class ClipFeatureExtractor(nn.Module):
    """Unified CLIP image/text feature wrapper."""

    _WIDTH = {
        "RN50": 1024,
        "ViT-L/14": 768,
        "ViT-B/32": 512,
        "RN50x64": 1024,
    }

    def __init__(
        self,
        variant: str,
        *,
        device: torch.device,
        normalize_embeddings: bool = True,
        use_hidden_state: bool = True,
    ) -> None:
        super().__init__()
        if variant not in self._WIDTH:
            raise ValueError(f"Unsupported CLIP variant: {variant}")

        self.variant = variant
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.use_hidden_state = use_hidden_state

        clip_model, _ = clip.load(variant, device=device)
        clip_model.eval()
        clip_model.requires_grad_(False)
        self._clip_model = clip_model

        if use_hidden_state:
            if variant != "ViT-L/14":
                raise ValueError("Hidden-state extraction is only supported for ViT-L/14")

            from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection

            self._vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
            self._text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval().to(device)
            self._tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

            self._vision_model.requires_grad_(False)
            self._text_model.requires_grad_(False)
        else:
            self._vision_model = None
            self._text_model = None
            self._tokenizer = None

        clip_size = (448, 448) if variant == "RN50x64" else (224, 224)
        mean = np.array([0.48145466, 0.4578275, 0.40821073])
        std = np.array([0.26862954, 0.26130258, 0.27577711])
        self._image_preprocess = transforms.Compose(
            [
                transforms.Resize(size=clip_size[0], interpolation=transforms.InterpolationMode.BICUBIC, antialias=None),
                transforms.CenterCrop(size=clip_size),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    @property
    def width(self) -> int:
        return self._WIDTH[self.variant]

    @property
    def image_output_dim(self) -> int:
        return 257 * self.width

    @property
    def text_output_dim(self) -> int:
        return 77 * self.width

    def _normalize_hidden_image(self, outputs) -> torch.Tensor:
        hidden = outputs.last_hidden_state
        hidden = self._vision_model.vision_model.post_layernorm(hidden)
        hidden = self._vision_model.visual_projection(hidden)
        if self.normalize_embeddings:
            hidden = hidden / torch.norm(hidden[:, 0], dim=-1, keepdim=True).unsqueeze(1)
        return hidden

    def _normalize_hidden_text(self, outputs) -> torch.Tensor:
        hidden = self._text_model.text_projection(outputs.last_hidden_state)
        pooled = outputs.text_embeds
        hidden = hidden / torch.norm(pooled.unsqueeze(1), dim=-1, keepdim=True)
        return hidden

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self._image_preprocess(images.to(self.device))
        if self.use_hidden_state:
            outputs = self._vision_model(images)
            return self._normalize_hidden_image(outputs)

        embeddings = self._clip_model.encode_image(images)
        if self.normalize_embeddings:
            embeddings = nn.functional.normalize(embeddings, dim=-1)
        return embeddings

    @torch.no_grad()
    def encode_texts(self, prompts: Sequence[str]) -> torch.Tensor:
        if not self.use_hidden_state:
            tokens = clip.tokenize(list(prompts), truncate=True).to(self.device)
            embeddings = self._clip_model.encode_text(tokens)
            if self.normalize_embeddings:
                embeddings = nn.functional.normalize(embeddings, dim=-1)
            return embeddings

        tokenized = self._tokenizer(
            list(prompts),
            padding="max_length",
            max_length=self._tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self._text_model(tokenized.input_ids.to(self.device))
        return self._normalize_hidden_text(outputs)
