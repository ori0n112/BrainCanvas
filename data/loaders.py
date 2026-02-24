from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import Architecture, DataConfig
from data.dataset import NSDRecordDataset


@dataclass(slots=True)
class SubjectLoaders:
    subject: int
    train: DataLoader
    val: DataLoader


@dataclass(slots=True)
class LoaderBundle:
    mode: str
    source_loaders: list[SubjectLoaders] = field(default_factory=list)
    target_loader: SubjectLoaders | None = None

    @property
    def is_single(self) -> bool:
        return self.mode == "single"

    @property
    def is_multi(self) -> bool:
        return self.mode == "multi"

    @property
    def is_adapt(self) -> bool:
        return self.mode == "adapt"


def _subject_split_paths(data_root: Path, subject: int) -> tuple[Path, Path]:
    train_dir = data_root / "webdataset_avg_split" / "train" / f"subj0{subject}"
    val_dir = data_root / "webdataset_avg_split" / "val" / f"subj0{subject}"
    return train_dir, val_dir


def _make_dataloader(
    path: Path,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    shuffle: bool,
    pool_size: int,
    pool_mode,
    dataset_length: int | None,
) -> DataLoader:
    dataset = NSDRecordDataset(
        path,
        pool_size=pool_size,
        pool_mode=pool_mode,
        length=dataset_length if shuffle else None,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        shuffle=shuffle,
        generator=generator,
    )


def _build_subject_loaders(cfg: DataConfig, subject: int, seed: int) -> SubjectLoaders:
    train_dir, val_dir = _subject_split_paths(cfg.data_root, subject)
    train_dl = _make_dataloader(
        train_dir,
        batch_size=cfg.batch_size,
        workers=cfg.num_workers,
        seed=seed,
        shuffle=True,
        pool_size=cfg.pool_size,
        pool_mode=cfg.pool_mode,
        dataset_length=cfg.dataset_length,
    )
    val_dl = _make_dataloader(
        val_dir,
        batch_size=cfg.val_batch_size,
        workers=cfg.num_workers,
        seed=seed,
        shuffle=False,
        pool_size=cfg.pool_size,
        pool_mode=cfg.pool_mode,
        dataset_length=None,
    )
    return SubjectLoaders(subject=subject, train=train_dl, val=val_dl)


def build_loader_bundle(cfg: DataConfig, architecture: Architecture, seed: int) -> LoaderBundle:
    if cfg.adapting:
        source = [_build_subject_loaders(cfg, subject, seed) for subject in cfg.source_subjects]
        target = _build_subject_loaders(cfg, cfg.target_subject, seed)
        return LoaderBundle(mode="adapt", source_loaders=source, target_loader=target)

    source = [_build_subject_loaders(cfg, subject, seed) for subject in cfg.subjects]
    mode = "single" if len(source) == 1 else "multi"

    # If the user chooses single-subject mode explicitly, keep only one subject loader.
    if architecture is Architecture.SINGLE:
        source = source[:1]
        mode = "single"

    return LoaderBundle(mode=mode, source_loaders=source)
