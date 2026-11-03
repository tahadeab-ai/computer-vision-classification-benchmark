from typing import Tuple
from torchvision import transforms
from torchvision.transforms import InterpolationMode

CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR100_STD = [0.2675, 0.2565, 0.2761]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(
    model_name: str = "custom_cnn",
    is_train: bool = True,
    image_size: Tuple[int, int] = (224, 224),
) -> transforms.Compose:
    """Generate architecture-specific preprocessing and augmentation pipelines.

    Args:
        model_name: Name of model ('custom_cnn', 'vgg16', 'mobilenet_v2', 'resnet50',
                    'efficientnet_b0', 'vit_b_16').
        is_train: Whether to include training data augmentation.
        image_size: Target image height and width.
    """
    if model_name == "custom_cnn":
        # Native CIFAR-100 resolution (32x32) with CIFAR-100 normalization
        if is_train:
            return transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CIFAR100_MEAN, std=CIFAR100_STD),
                ]
            )
    else:
        # Pretrained backbones: Upscaled to 224x224 with ImageNet normalization
        interpolation = (
            InterpolationMode.BICUBIC
            if "vit" in model_name or "efficientnet" in model_name
            else InterpolationMode.BILINEAR
        )

        if is_train:
            return transforms.Compose(
                [
                    transforms.Resize(image_size, interpolation=interpolation),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(image_size, interpolation=interpolation),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ]
            )
