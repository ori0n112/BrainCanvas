from __future__ import annotations

from pathlib import Path

import torch


class CheckpointManager:
    def __init__(self, run_dir: Path, accelerator) -> None:
        self.run_dir = run_dir
        self.accelerator = accelerator
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save(self, *, tag: str, epoch: int, model: torch.nn.Module) -> None:
        state_dir = self.run_dir / tag
        model_path = self.run_dir / f"{tag}.pt"

        if self.accelerator.is_main_process:
            payload = {
                "epoch": epoch,
                "model_state_dict": self.accelerator.unwrap_model(model).state_dict(),
            }
            torch.save(payload, model_path)

        self.accelerator.save_state(str(state_dir))

    def load_weights(self, model: torch.nn.Module, checkpoint_path: str | Path) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu")
        self.accelerator.unwrap_model(model).load_state_dict(payload["model_state_dict"], strict=False)

    def resume_training_state(self) -> int:
        state_dir = self.run_dir / "last"
        model_path = self.run_dir / "last.pt"

        self.accelerator.load_state(str(state_dir))
        payload = torch.load(model_path, map_location="cpu")
        return int(payload["epoch"])
