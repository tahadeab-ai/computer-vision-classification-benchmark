import os
import tarfile
import urllib.request
from typing import Any, Callable, Dict, List, Optional
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import CIFAR100

CIFAR100_FAST_MIRROR = "https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz"


def ensure_cifar100_downloaded(data_dir: str = "./data") -> None:
    """Download CIFAR-100 from high-speed mirror if not present or incomplete."""
    os.makedirs(data_dir, exist_ok=True)
    extracted_dir = os.path.join(data_dir, "cifar-100-python")
    train_file = os.path.join(extracted_dir, "train")
    test_file = os.path.join(extracted_dir, "test")

    if os.path.exists(train_file) and os.path.exists(test_file):
        return

    archive_path = os.path.join(data_dir, "cifar-100-python.tar.gz")
    if not os.path.exists(archive_path) or os.path.getsize(archive_path) < 100 * 1024 * 1024:
        print(f"[Dataset] Downloading CIFAR-100 archive from high-speed CDN: {CIFAR100_FAST_MIRROR}")
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(CIFAR100_FAST_MIRROR, archive_path)

    print(f"[Dataset] Extracting CIFAR-100 archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print(f"[Dataset] CIFAR-100 successfully extracted to {extracted_dir}")

CIFAR100_FINE_CLASSES: List[str] = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]


class CIFAR100DatasetWrapper:
    """Manager for downloading, verifying, and packaging CIFAR-100."""

    def __init__(self, data_dir: str = "./data") -> None:
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def download_and_verify(self) -> Dict[str, Any]:
        """Download CIFAR-100 train and test sets, verify integrity and return summary."""
        ensure_cifar100_downloaded(self.data_dir)
        train_raw = CIFAR100(root=self.data_dir, train=True, download=False)
        test_raw = CIFAR100(root=self.data_dir, train=False, download=False)

        train_data = train_raw.data
        train_targets = np.array(train_raw.targets)
        test_data = test_raw.data
        test_targets = np.array(test_raw.targets)

        # Integrity checks
        assert len(train_data) == 50000, f"Expected 50000 train samples, got {len(train_data)}"
        assert len(test_data) == 10000, f"Expected 10000 test samples, got {len(test_data)}"
        assert train_data.shape[1:] == (32, 32, 3), f"Expected shape (32, 32, 3), got {train_data.shape[1:]}"
        assert len(np.unique(train_targets)) == 100, f"Expected 100 unique classes, got {len(np.unique(train_targets))}"

        # Class distribution check (CIFAR-100 is perfectly balanced: 500 train, 100 test per class)
        train_counts = np.bincount(train_targets, minlength=100)
        test_counts = np.bincount(test_targets, minlength=100)
        is_train_balanced = bool(np.all(train_counts == 500))
        is_test_balanced = bool(np.all(test_counts == 100))

        # Check for NaN / corrupted byte ranges
        has_nans = bool(np.isnan(train_data).any() or np.isnan(test_data).any())

        return {
            "train_samples": len(train_data),
            "test_samples": len(test_data),
            "image_shape": list(train_data.shape[1:]),
            "num_classes": 100,
            "train_balanced": is_train_balanced,
            "test_balanced": is_test_balanced,
            "corrupted_or_nan": has_nans,
            "classes": CIFAR100_FINE_CLASSES,
        }


class TransformedSubset(torch.utils.data.Dataset):
    """Applies specific torchvision transforms dynamically to an underlying subset."""

    def __init__(
        self,
        base_dataset: CIFAR100,
        indices: List[int],
        transform: Optional[Callable] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        true_idx = self.indices[idx]
        img_arr, target = self.base_dataset.data[true_idx], self.base_dataset.targets[true_idx]
        img = Image.fromarray(img_arr)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
