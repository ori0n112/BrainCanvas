from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import cycle
from typing import Iterator

import torch

from data.loaders import LoaderBundle, SubjectLoaders


@dataclass(slots=True)
class Batch:
    voxel: torch.Tensor
    image: torch.Tensor
    coco_id: torch.Tensor
    subject_id: torch.Tensor

    def training_view(self, repeat_index: int) -> "Batch":
        voxel = self.voxel
        if voxel.dim() >= 3:
            idx = repeat_index % voxel.shape[1]
            voxel = voxel[:, idx, ...]
        return replace(self, voxel=voxel)

    def validation_view(self) -> "Batch":
        voxel = self.voxel
        if voxel.dim() >= 3:
            voxel = voxel.mean(dim=1)
        return replace(self, voxel=voxel)


def _from_loader_item(item: dict) -> Batch:
    return Batch(
        voxel=item["voxel"],
        image=item["image"],
        coco_id=item["coco_id"],
        subject_id=item["subject_id"].long().view(-1),
    )


def _merge_batches(batches: list[Batch]) -> Batch:
    return Batch(
        voxel=torch.cat([b.voxel for b in batches], dim=0),
        image=torch.cat([b.image for b in batches], dim=0),
        coco_id=torch.cat([b.coco_id for b in batches], dim=0),
        subject_id=torch.cat([b.subject_id for b in batches], dim=0),
    )


class BatchStream:
    def prepare(self, accelerator) -> "BatchStream":
        raise NotImplementedError

    def train_batches(self) -> Iterator[Batch]:
        raise NotImplementedError

    def val_batches(self) -> Iterator[Batch]:
        raise NotImplementedError

    @property
    def train_steps(self) -> int:
        raise NotImplementedError


class SingleBatchStream(BatchStream):
    def __init__(self, loader: SubjectLoaders) -> None:
        self.loader = loader

    def prepare(self, accelerator) -> "SingleBatchStream":
        self.loader.train = accelerator.prepare(self.loader.train)
        self.loader.val = accelerator.prepare(self.loader.val)
        return self

    def train_batches(self) -> Iterator[Batch]:
        for item in self.loader.train:
            yield _from_loader_item(item)

    def val_batches(self) -> Iterator[Batch]:
        for item in self.loader.val:
            yield _from_loader_item(item)

    @property
    def train_steps(self) -> int:
        return len(self.loader.train)


class MultiBatchStream(BatchStream):
    def __init__(self, loaders: list[SubjectLoaders]) -> None:
        self.loaders = loaders

    def prepare(self, accelerator) -> "MultiBatchStream":
        for loader in self.loaders:
            loader.train = accelerator.prepare(loader.train)
            loader.val = accelerator.prepare(loader.val)
        return self

    def train_batches(self) -> Iterator[Batch]:
        for items in zip(*[loader.train for loader in self.loaders]):
            batches = [_from_loader_item(item) for item in items]
            yield _merge_batches(batches)

    def val_batches(self) -> Iterator[Batch]:
        for items in zip(*[loader.val for loader in self.loaders]):
            batches = [_from_loader_item(item) for item in items]
            yield _merge_batches(batches)

    @property
    def train_steps(self) -> int:
        return len(self.loaders[0].train)


class AdaptBatchStream(BatchStream):
    def __init__(self, source_loaders: list[SubjectLoaders], target_loader: SubjectLoaders) -> None:
        self.source_loaders = source_loaders
        self.target_loader = target_loader

    def prepare(self, accelerator) -> "AdaptBatchStream":
        self.target_loader.train = accelerator.prepare(self.target_loader.train)
        self.target_loader.val = accelerator.prepare(self.target_loader.val)
        for loader in self.source_loaders:
            loader.train = accelerator.prepare(loader.train)
            loader.val = accelerator.prepare(loader.val)
        return self

    def train_batches(self) -> Iterator[Batch]:
        source_iters = [cycle(loader.train) for loader in self.source_loaders]
        for step, target_item in enumerate(self.target_loader.train):
            source_item = next(source_iters[step % len(source_iters)])
            merged = _merge_batches([_from_loader_item(source_item), _from_loader_item(target_item)])
            yield merged

    def val_batches(self) -> Iterator[Batch]:
        source_iters = [cycle(loader.val) for loader in self.source_loaders]
        for step, target_item in enumerate(self.target_loader.val):
            source_item = next(source_iters[step % len(source_iters)])
            merged = _merge_batches([_from_loader_item(source_item), _from_loader_item(target_item)])
            yield merged

    @property
    def train_steps(self) -> int:
        return len(self.target_loader.train)


def create_batch_stream(bundle: LoaderBundle) -> BatchStream:
    if bundle.is_adapt:
        assert bundle.target_loader is not None
        return AdaptBatchStream(bundle.source_loaders, bundle.target_loader)
    if bundle.is_single:
        return SingleBatchStream(bundle.source_loaders[0])
    return MultiBatchStream(bundle.source_loaders)
