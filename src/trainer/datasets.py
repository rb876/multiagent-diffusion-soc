from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.datasets.vision import VisionDataset


DEFAULT_DATA_ROOT = Path("data")
DEFAULT_NUM_WORKERS = 4


class DatasetName(str, Enum):
    MNIST = "MNIST"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    TEST = "test"


@dataclass(frozen=True)
class DatasetConfig:
    factory: Callable[[Path, bool, bool, Callable], VisionDataset]
    default_transform: Callable = transforms.ToTensor


@dataclass(frozen=True)
class DatasetLoaders:
    train: DataLoader
    test: DataLoader


def _mnist_factory(root: Path, train: bool, download: bool, transform: Callable) -> VisionDataset:
    return MNIST(
        root=str(root),
        train=train,
        download=download,
        transform=transform(),
    )


_DATASET_REGISTRY: Dict[DatasetName, DatasetConfig] = {
    DatasetName.MNIST: DatasetConfig(factory=_mnist_factory),
}


def _resolve_dataset_name(name: Union[str, DatasetName]) -> DatasetName:
    if isinstance(name, DatasetName):
        return name
    normalized = str(name).upper()
    for candidate in DatasetName:
        if candidate.value.upper() == normalized:
            return candidate
    available = ", ".join(item.value for item in DatasetName)
    raise ValueError(f"Dataset {name!r} not recognized. Available: {available}.")


def _resolve_dataset_split(split: Union[str, DatasetSplit]) -> DatasetSplit:
    if isinstance(split, DatasetSplit):
        return split
    normalized = str(split).lower()
    for candidate in DatasetSplit:
        if candidate.value == normalized:
            return candidate
    available = ", ".join(item.value for item in DatasetSplit)
    raise ValueError(f"Dataset split {split!r} not recognized. Available: {available}.")


def get_dataset_loader(
    name: Union[str, DatasetName],
    batch_size: int,
    *,
    train: bool = True,
    split: Optional[Union[str, DatasetSplit]] = None,
    root: Union[str, Path] = DEFAULT_DATA_ROOT,
    num_workers: int = DEFAULT_NUM_WORKERS,
    shuffle: Optional[bool] = None,
    download: bool = True,
    transform: Optional[Callable] = None,
    persistent_workers: Optional[bool] = None,
    **dataloader_kwargs: Any,
) -> DataLoader:
    dataset_name = _resolve_dataset_name(name)
    dataset_split = (
        _resolve_dataset_split(split) if split is not None else (DatasetSplit.TRAIN if train else DatasetSplit.TEST)
    )
    config = _DATASET_REGISTRY[dataset_name]

    data_root = Path(root)
    data_root.mkdir(parents=True, exist_ok=True)

    dataset_transform = transform or config.default_transform
    dataset = config.factory(
        data_root,
        dataset_split is DatasetSplit.TRAIN,
        download,
        dataset_transform,
    )

    effective_shuffle = shuffle if shuffle is not None else dataset_split is DatasetSplit.TRAIN
    effective_persistent_workers = (
        persistent_workers if persistent_workers is not None else num_workers > 0
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        num_workers=num_workers,
        persistent_workers=effective_persistent_workers if num_workers > 0 else False,
        **dataloader_kwargs,
    )


def get_dataset_loaders(
    name: Union[str, DatasetName],
    train_batch_size: int,
    test_batch_size: Optional[int] = None,
    *,
    root: Union[str, Path] = DEFAULT_DATA_ROOT,
    num_workers: int = DEFAULT_NUM_WORKERS,
    download: bool = True,
    train_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    train_shuffle: Optional[bool] = None,
    test_shuffle: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    train_loader_kwargs: Optional[Dict[str, Any]] = None,
    test_loader_kwargs: Optional[Dict[str, Any]] = None,
) -> DatasetLoaders:
    train_loader = get_dataset_loader(
        name,
        train_batch_size,
        split=DatasetSplit.TRAIN,
        root=root,
        num_workers=num_workers,
        shuffle=train_shuffle,
        download=download,
        transform=train_transform,
        persistent_workers=persistent_workers,
        **(train_loader_kwargs or {}),
    )
    test_loader = get_dataset_loader(
        name,
        test_batch_size or train_batch_size,
        split=DatasetSplit.TEST,
        root=root,
        num_workers=num_workers,
        shuffle=test_shuffle if test_shuffle is not None else False,
        download=download,
        transform=test_transform,
        persistent_workers=persistent_workers,
        **(test_loader_kwargs or {}),
    )
    return DatasetLoaders(train=train_loader, test=test_loader)


def get_MNIST_loader(batch_size: int, **kwargs: Any) -> DataLoader:
    return get_dataset_loader(DatasetName.MNIST, batch_size=batch_size, **kwargs)


def get_MNIST_loaders(train_batch_size: int, test_batch_size: Optional[int] = None, **kwargs: Any) -> DatasetLoaders:
    return get_dataset_loaders(
        DatasetName.MNIST,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        **kwargs,
    )


def get_dataloader_by_name(name: Union[str, DatasetName], batch_size: int, **kwargs: Any) -> DataLoader:
    return get_dataset_loader(name, batch_size, **kwargs)