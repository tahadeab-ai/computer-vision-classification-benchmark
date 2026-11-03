import pytest
import numpy as np
from cv_benchmark.data.dataset import CIFAR100_FINE_CLASSES, CIFAR100DatasetWrapper
from cv_benchmark.data.loaders import get_train_val_indices


def test_fine_classes_count():
    assert len(CIFAR100_FINE_CLASSES) == 100
    assert "apple" in CIFAR100_FINE_CLASSES
    assert "worm" in CIFAR100_FINE_CLASSES


def test_train_val_split_integrity():
    # 50,000 samples, 100 classes with 500 each
    mock_targets = np.repeat(np.arange(100), 500)
    train_idx, val_idx = get_train_val_indices(mock_targets, val_ratio=0.1, seed=42)

    assert len(train_idx) == 45000
    assert len(val_idx) == 5000

    # Ensure zero overlap (no data leakage)
    intersection = np.intersect1d(train_idx, val_idx)
    assert len(intersection) == 0

    # Ensure class balance is preserved in both splits
    train_counts = np.bincount(mock_targets[train_idx])
    val_counts = np.bincount(mock_targets[val_idx])
    assert np.all(train_counts == 450)
    assert np.all(val_counts == 50)
