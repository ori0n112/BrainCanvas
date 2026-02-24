from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from config import PoolMode

try:
    import kornia
    from kornia.augmentation.container import AugmentationSequential

    _AUGMENTOR = AugmentationSequential(
        kornia.augmentation.RandomResizedCrop((224, 224), (0.8, 1.0), p=0.3),
        kornia.augmentation.Resize((224, 224)),
        kornia.augmentation.RandomBrightness(brightness=(0.8, 1.2), clip_output=True, p=0.2),
        kornia.augmentation.RandomContrast(contrast=(0.8, 1.2), clip_output=True, p=0.2),
        kornia.augmentation.RandomGamma((0.8, 1.2), (1.0, 1.3), p=0.2),
        kornia.augmentation.RandomSaturation((0.8, 1.2), p=0.2),
        kornia.augmentation.RandomHue((-0.1, 0.1), p=0.2),
        kornia.augmentation.RandomSharpness((0.8, 1.2), p=0.2),
        kornia.augmentation.RandomGrayscale(p=0.2),
        data_keys=["input"],
    )
except Exception:
    _AUGMENTOR = None


@dataclass(slots=True)
class SampleRecord:
    key: str
    subject: int
    image_path: Path
    voxel_path: Path
    coco_path: Path


def pool_voxels(voxels: torch.Tensor, pool_size: int, mode: PoolMode) -> torch.Tensor:
    voxels = voxels.float()
    if mode is PoolMode.AVG:
        return nn.AdaptiveAvgPool1d(pool_size)(voxels)
    if mode is PoolMode.MAX:
        return nn.AdaptiveMaxPool1d(pool_size)(voxels)
    if mode is PoolMode.RESIZE:
        resized = F.interpolate(voxels.unsqueeze(1), size=pool_size, mode="linear", align_corners=False)
        return resized.squeeze(1)
    raise ValueError(f"Unsupported pool mode: {mode}")


def maybe_augment_images(images: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled or _AUGMENTOR is None:
        return images
    return _AUGMENTOR(images)


class NSDRecordDataset(Dataset[dict[str, Any]]):
    """Lightweight NSD dataset reader.

    Expected per-sample files in one folder:
    - `<sample_id>.jpg`
    - `<sample_id>.nsdgeneral.npy`
    - `<sample_id>.coco73k.npy`
    """

    def __init__(
        self,
        root: Path,
        *,
        pool_size: int,
        pool_mode: PoolMode,
        length: int | None = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self._records = self._discover_records(root)

        if length is not None:
            if length == 0:
                raise ValueError("length cannot be zero")
            if length > 0:
                self._records = self._records[:length]
            else:
                self._records = self._records[length:]

        if len(self._records) == 0:
            raise RuntimeError(f"No valid samples found under: {root}")

    @staticmethod
    def _extract_subject_id(path: Path) -> int:
        for part in path.parts:
            if part.startswith("subj"):
                value = part.replace("subj", "")
                if value.isdigit():
                    return int(value)
        return 0

    @classmethod
    def _discover_records(cls, root: Path) -> list[SampleRecord]:
        files = sorted(root.iterdir())
        buckets: dict[str, dict[str, Path]] = {}

        for file_path in files:
            if not file_path.is_file() or "." not in file_path.name:
                continue
            sample_id, ext = file_path.name.split(".", maxsplit=1)
            slots = buckets.setdefault(sample_id, {})
            slots[ext] = file_path

        subject_id = cls._extract_subject_id(root)
        records: list[SampleRecord] = []
        for key, paths in sorted(buckets.items()):
            if "jpg" not in paths or "nsdgeneral.npy" not in paths or "coco73k.npy" not in paths:
                continue
            records.append(
                SampleRecord(
                    key=key,
                    subject=subject_id,
                    image_path=paths["jpg"],
                    voxel_path=paths["nsdgeneral.npy"],
                    coco_path=paths["coco73k.npy"],
                )
            )
        return records

    @staticmethod
    def _load_rgb_image(image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1))

    @staticmethod
    def _load_npy(path: Path) -> torch.Tensor:
        return torch.from_numpy(np.load(path))

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self._records[index % len(self._records)]

        voxel = self._load_npy(record.voxel_path)
        voxel = pool_voxels(voxel, self.pool_size, self.pool_mode)

        return {
            "voxel": voxel,
            "image": self._load_rgb_image(record.image_path),
            "coco_id": self._load_npy(record.coco_path),
            "subject_id": torch.tensor(record.subject, dtype=torch.long),
            "sample_key": record.key,
        }
