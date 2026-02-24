from __future__ import annotations

import abc
import math
from typing import Callable

import torch
from diffusers.models.attention_processor import Attention


class AttentionController(abc.ABC):
    """Base interface for intercepting UNet attention maps."""

    def __init__(self) -> None:
        self.current_step = 0
        self.current_attention_layer = 0
        self.num_attention_layers = -1

    def reset(self) -> None:
        self.current_step = 0
        self.current_attention_layer = 0

    def step_callback(self, latent: torch.Tensor) -> torch.Tensor:
        return latent

    def between_steps(self) -> None:
        return

    @abc.abstractmethod
    def forward(self, attn: torch.Tensor, heads: int, is_cross: bool, location: str) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, attn: torch.Tensor, heads: int, is_cross: bool, location: str) -> torch.Tensor:
        if is_cross:
            attn = self.forward(attn, heads, is_cross, location)
        else:
            half = attn.shape[0] // 2
            attn[half:] = self.forward(attn[half:], heads, is_cross, location)

        self.current_attention_layer += 1
        if self.current_attention_layer == self.num_attention_layers:
            self.current_attention_layer = 0
            self.between_steps()
            self.current_step += 1

        return attn


class AttentionStoreClassPrompts(AttentionController):
    """Stores self/cross attention maps across a step range."""

    @staticmethod
    def _empty_store() -> dict[str, list[torch.Tensor]]:
        return {
            "down_cross": [],
            "mid_cross": [],
            "up_cross": [],
            "down_self": [],
            "mid_self": [],
            "up_self": [],
        }

    def __init__(self, *, start_step: int = 0, end_step: int = 1000) -> None:
        super().__init__()
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.step_store = self._empty_store()
        self.accumulated_store: dict[str, list[torch.Tensor]] | None = None

    def reset(self) -> None:
        super().reset()
        self.step_store = self._empty_store()
        self.accumulated_store = None

    def forward(self, attn: torch.Tensor, heads: int, is_cross: bool, location: str) -> torch.Tensor:
        if self.start_step <= self.current_step <= self.end_step:
            if attn.shape[1] <= 64**2:
                spatial_res = int(math.sqrt(attn.shape[1]))
                mapped = attn.reshape(-1, heads, spatial_res, spatial_res, attn.shape[-1])
                key = f"{location}_{'cross' if is_cross else 'self'}"
                self.step_store[key].append(mapped)
        return attn

    def between_steps(self) -> None:
        if not (self.start_step <= self.current_step <= self.end_step):
            return

        if self.accumulated_store is None:
            self.accumulated_store = self.step_store
            self.step_store = self._empty_store()
            return

        for key in self.accumulated_store:
            for idx in range(len(self.accumulated_store[key])):
                self.accumulated_store[key][idx] += self.step_store[key][idx]
        self.step_store = self._empty_store()

    def average_attention(self) -> dict[str, list[torch.Tensor]]:
        if self.accumulated_store is None:
            return self._empty_store()

        start = max(0, self.start_step)
        end = min(self.current_step, self.end_step + 1)
        denominator = max(1, end - start)
        return {
            key: [item / denominator for item in self.accumulated_store[key]] for key in self.accumulated_store
        }


class StoredAttnClassPromptsProcessor:
    """Attention processor that sends maps into an AttentionStoreClassPrompts."""

    def __init__(self, *, attn_store: AttentionStoreClassPrompts, location: str) -> None:
        self.attn_store = attn_store
        self.location = location

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        query = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(context)
        value = attn.to_v(context)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        stored_cross = False
        if is_cross:
            primary_length = attn.heads * batch_size
            cross_key = key[primary_length:]
            cross_query = query[primary_length // 2 :]
            key = key[:primary_length]
            value = value[:primary_length]
            if cross_key.shape[0] > 0 and cross_query.shape[0] > 0:
                cross_probs = attn.get_attention_scores(cross_query, cross_key, None)
                self.attn_store(cross_probs, attn.heads, True, self.location)
                stored_cross = True

        probs = attn.get_attention_scores(query, key, attention_mask)
        if is_cross and not stored_cross:
            self.attn_store(probs, attn.heads, True, self.location)
        if not is_cross:
            self.attn_store(probs, attn.heads, False, self.location)

        hidden_states = torch.bmm(probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def aggregate_attention_maps(
    attention_store: AttentionStoreClassPrompts,
    *,
    resolution: int,
    is_cross: bool,
    locations: tuple[str, ...] = ("up", "down", "mid"),
) -> torch.Tensor:
    maps = []
    averaged = attention_store.average_attention()
    key_suffix = "cross" if is_cross else "self"

    for location in locations:
        key = f"{location}_{key_suffix}"
        for item in averaged.get(key, []):
            if item.shape[2] == resolution:
                maps.append(item)

    if len(maps) == 0:
        raise RuntimeError(
            f"No attention maps collected for resolution={resolution}, is_cross={is_cross}, locations={locations}"
        )
    return torch.cat(maps, dim=1).mean(dim=1)


def register_attention_controller(
    pipeline,
    controller: AttentionStoreClassPrompts,
    processor_factory: Callable[..., StoredAttnClassPromptsProcessor] = StoredAttnClassPromptsProcessor,
) -> None:
    if not hasattr(pipeline, "unet"):
        raise AttributeError("pipeline must provide `unet`")
    if not hasattr(pipeline.unet, "attn_processors"):
        raise AttributeError("pipeline.unet must provide `attn_processors`")

    attn_processors = {}
    count = 0
    for name in pipeline.unet.attn_processors.keys():
        if name.startswith("mid_block"):
            location = "mid"
        elif name.startswith("up_blocks"):
            location = "up"
        elif name.startswith("down_blocks"):
            location = "down"
        else:
            continue

        count += 1
        attn_processors[name] = processor_factory(attn_store=controller, location=location)

    pipeline.unet.set_attn_processor(attn_processors)
    controller.num_attention_layers = count
