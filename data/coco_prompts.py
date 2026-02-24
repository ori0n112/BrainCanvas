from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch


class COCOTextRepository:
    """Loads NSD-to-COCO mappings and produces training prompts."""

    def __init__(
        self,
        data_root: Path,
        *,
        append_categories: bool,
        delimiter: str,
        max_categories: int,
    ) -> None:
        self._append_categories = append_categories
        self._delimiter = delimiter
        self._max_categories = max_categories

        mapping_path = data_root / "nsddata" / "experiments" / "nsd" / "nsd_stim_info_merged.csv"
        annotations_dir = data_root / "nsddata_stimuli" / "stimuli" / "nsd" / "annotations"

        stim_info = pd.read_csv(mapping_path)
        self._splits = stim_info["cocoSplit"].astype(str).tolist()
        self._coco_ids = stim_info["cocoId"].astype(int).tolist()

        self._captions = {
            "train2017": self._load_caption_annotations(annotations_dir / "captions_train2017.json"),
            "val2017": self._load_caption_annotations(annotations_dir / "captions_val2017.json"),
        }
        self._categories = {
            "train2017": self._load_instance_categories(annotations_dir / "instances_train2017.json"),
            "val2017": self._load_instance_categories(annotations_dir / "instances_val2017.json"),
        }

    @staticmethod
    def _load_caption_annotations(path: Path) -> dict[int, list[str]]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        by_image_id: dict[int, list[str]] = defaultdict(list)
        for ann in payload["annotations"]:
            by_image_id[int(ann["image_id"])].append(str(ann["caption"]).strip())
        return by_image_id

    @staticmethod
    def _load_instance_categories(path: Path) -> dict[int, list[str]]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        category_name = {int(cat["id"]): str(cat["name"]).strip() for cat in payload["categories"]}
        by_image_id: dict[int, set[str]] = defaultdict(set)
        for ann in payload["annotations"]:
            image_id = int(ann["image_id"])
            category_id = int(ann["category_id"])
            if category_id in category_name:
                by_image_id[image_id].add(category_name[category_id])

        ordered: dict[int, list[str]] = {}
        for key, values in by_image_id.items():
            ordered[key] = sorted(values)
        return ordered

    @staticmethod
    def _to_python_list(values: torch.Tensor | np.ndarray | Iterable[int]) -> list:
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().tolist()
        if isinstance(values, np.ndarray):
            return values.tolist()
        return list(values)

    @staticmethod
    def _extract_nsd_index(raw_value, repeat_index: int) -> int:
        if isinstance(raw_value, (list, tuple)):
            if len(raw_value) == 0:
                return 0
            idx = min(repeat_index, len(raw_value) - 1)
            return int(raw_value[idx])
        if isinstance(raw_value, np.ndarray):
            flat = raw_value.reshape(-1)
            idx = min(repeat_index, len(flat) - 1)
            return int(flat[idx])
        return int(raw_value)

    def _resolve_caption(self, nsd_index: int, repeat_index: int) -> str:
        split = self._splits[nsd_index]
        coco_id = self._coco_ids[nsd_index]

        candidates = self._captions.get(split, {}).get(coco_id, [""])
        caption = candidates[min(repeat_index, len(candidates) - 1)] if candidates else ""

        if not self._append_categories:
            return caption

        categories = self._categories.get(split, {}).get(coco_id, [])
        if len(categories) == 0:
            return caption

        if self._max_categories > 0:
            categories = categories[: self._max_categories]
        return f"{caption}{self._delimiter}{', '.join(categories)}"

    def prompts_for_batch(self, coco_ids, repeat_index: int) -> list[str]:
        values = self._to_python_list(coco_ids)
        return [self._resolve_caption(self._extract_nsd_index(value, repeat_index), repeat_index) for value in values]
