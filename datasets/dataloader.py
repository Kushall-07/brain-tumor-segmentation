
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from monai.data import list_data_collate
from monai.transforms import (
    CastToTyped,
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    RandFlipd,
    RandGaussianNoised,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    SpatialPadd,
)

from datasets.brats_dataset import BraTSDataset


class _BraTSAsDict(Dataset):
    def __init__(self, base: Dataset, transforms: Optional[Callable] = None) -> None:
        self.base = base
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        sample = self.base[idx]

        if isinstance(sample, (tuple, list)):
            if len(sample) == 3:
                image, mask, case_id = sample
            elif len(sample) == 2:
                image, mask = sample
                case_id = str(idx)
            else:
                raise ValueError(f"Unexpected sample tuple length: {len(sample)}")
        else:
            raise ValueError(f"Unexpected sample type: {type(sample)}")

        data = {"image": image, "label": mask, "case_id": case_id}
        if self.transforms is not None:
            data = self.transforms(data)
        return data


def get_train_transforms(patch_size: tuple[int, int, int] = (96, 96, 96)) -> Compose:
    return Compose(
        [
            EnsureChannelFirstd(keys=["image"], channel_dim=0),
            EnsureTyped(keys=["image", "label"]),
            CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            RandSpatialCropd(keys=["image", "label"], roi_size=patch_size, random_size=False),
            RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=2, prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
        ]
    )


def get_val_transforms(patch_size: tuple[int, int, int] = (96, 96, 96)) -> Compose:
    return Compose(
        [
            EnsureChannelFirstd(keys=["image"], channel_dim=0),
            EnsureTyped(keys=["image", "label"]),
            CastToTyped(keys=["image", "label"], dtype=(torch.float32, torch.int64)),
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
            CenterSpatialCropd(keys=["image", "label"], roi_size=patch_size),
        ]
    )


def get_train_loader(
    root_dir: str | Path,
    case_dirs: Optional[Sequence[str | Path]] = None,
    batch_size: int = 1,
    num_workers: int = 2,
    pin_memory: bool = True,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    shuffle: bool = True,
) -> DataLoader:
    base_ds = BraTSDataset(root_dir=root_dir, case_dirs=case_dirs)
    ds = _BraTSAsDict(base=base_ds, transforms=get_train_transforms(patch_size=patch_size))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=list_data_collate,
    )


def get_val_loader(
    root_dir: str | Path,
    case_dirs: Optional[Sequence[str | Path]] = None,
    batch_size: int = 1,
    num_workers: int = 2,
    pin_memory: bool = True,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    shuffle: bool = False,
) -> DataLoader:
    base_ds = BraTSDataset(root_dir=root_dir, case_dirs=case_dirs)
    ds = _BraTSAsDict(base=base_ds, transforms=get_val_transforms(patch_size=patch_size))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=list_data_collate,
    )

