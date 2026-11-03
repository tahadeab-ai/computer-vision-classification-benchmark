from typing import Dict, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from sklearn.model_selection import StratifiedShuffleSplit

from .dataset import TransformedSubset, ensure_cifar100_downloaded
from .transforms import get_transforms


def get_train_val_indices(
    targets: np.ndarray,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform deterministic stratified train/val split ensuring zero data leakage."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))
    return train_idx, val_idx


def get_dataloaders(
    model_name: str = "custom_cnn",
    data_dir: str = "./data",
    batch_size: int = 64,
    val_ratio: float = 0.1,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = 42,
    image_size: Tuple[int, int] = (224, 224),
) -> Dict[str, DataLoader]:
    """Build train, validation, and test DataLoaders with model-specific transforms."""
    # Ensure dataset is downloaded from high-speed mirror
    ensure_cifar100_downloaded(data_dir)
    raw_train_dataset = CIFAR100(root=data_dir, train=True, download=False)
    raw_test_dataset = CIFAR100(root=data_dir, train=False, download=False)

    # Deterministic stratified split
    targets = np.array(raw_train_dataset.targets)
    train_indices, val_indices = get_train_val_indices(targets, val_ratio=val_ratio, seed=seed)
    test_indices = np.arange(len(raw_test_dataset))

    # Architecture-specific transforms
    train_tf = get_transforms(model_name=model_name, is_train=True, image_size=image_size)
    eval_tf = get_transforms(model_name=model_name, is_train=False, image_size=image_size)

    train_subset = TransformedSubset(raw_train_dataset, train_indices, transform=train_tf)
    val_subset = TransformedSubset(raw_train_dataset, val_indices, transform=eval_tf)
    test_subset = TransformedSubset(raw_test_dataset, test_indices, transform=eval_tf)

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def get_smoke_test_dataloaders(
    model_name: str = "custom_cnn",
    data_dir: str = "./data",
    batch_size: int = 16,
    train_size: int = 128,
    val_size: int = 64,
    test_size: int = 64,
    seed: int = 42,
    image_size: Tuple[int, int] = (224, 224),
) -> Dict[str, DataLoader]:
    """Lightweight dataloader subset for immediate pipeline and smoke testing."""
    ensure_cifar100_downloaded(data_dir)
    raw_train_dataset = CIFAR100(root=data_dir, train=True, download=False)
    raw_test_dataset = CIFAR100(root=data_dir, train=False, download=False)

    np.random.seed(seed)
    train_indices = np.random.choice(len(raw_train_dataset), size=train_size, replace=False)
    val_indices = np.random.choice(len(raw_train_dataset), size=val_size, replace=False)
    test_indices = np.random.choice(len(raw_test_dataset), size=test_size, replace=False)

    train_tf = get_transforms(model_name=model_name, is_train=True, image_size=image_size)
    eval_tf = get_transforms(model_name=model_name, is_train=False, image_size=image_size)

    train_subset = TransformedSubset(raw_train_dataset, train_indices, transform=train_tf)
    val_subset = TransformedSubset(raw_train_dataset, val_indices, transform=eval_tf)
    test_subset = TransformedSubset(raw_test_dataset, test_indices, transform=eval_tf)

    return {
        "train": DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False),
        "val": DataLoader(val_subset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_subset, batch_size=batch_size, shuffle=False),
    }
