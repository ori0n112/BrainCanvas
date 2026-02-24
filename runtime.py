from __future__ import annotations

import os

from accelerate import Accelerator, DeepSpeedPlugin

from config import RuntimeConfig


def create_accelerator(cfg: RuntimeConfig) -> Accelerator:
    plugin = DeepSpeedPlugin(
        zero_stage=cfg.deepspeed_zero_stage,
        gradient_clipping=cfg.gradient_clip,
    )
    accelerator = Accelerator(
        split_batches=False,
        mixed_precision=cfg.mixed_precision,
        deepspeed_plugin=plugin,
    )
    accelerator.print(f"PID={os.getpid()}")
    accelerator.print(f"Device={accelerator.device}")
    accelerator.print(accelerator.state)
    return accelerator
