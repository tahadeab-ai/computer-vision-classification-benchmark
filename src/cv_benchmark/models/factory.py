from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torchvision import models

from .custom_cnn import CustomCNN


def freeze_backbone(model: nn.Module, head_module_names: List[str]) -> None:
    """Freeze all layers in the model except those whose names match head_module_names."""
    for name, param in model.named_parameters():
        is_head = any(head_name in name for head_name in head_module_names)
        param.requires_grad = is_head


def get_parameter_groups(
    model: nn.Module,
    head_module_names: List[str],
    backbone_lr: float,
    head_lr: float,
    weight_decay: float = 1e-4,
) -> List[Dict]:
    """Separate backbone and classification head parameters for differential learning rates."""
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = any(head_name in name for head_name in head_module_names)
        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)

    groups = []
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
    return groups


def create_model(
    model_name: str = "custom_cnn",
    num_classes: int = 100,
    pretrained: bool = True,
    mode: str = "feature_extraction",  # 'scratch', 'feature_extraction', 'fine_tuning'
    dropout_rate: float = 0.3,
) -> Tuple[nn.Module, List[str]]:
    """Factory to instantiate and adapt any of the 6 CV benchmark architectures.

    Returns:
        Tuple of (model, head_module_names)
    """
    model_name = model_name.lower()

    if model_name == "custom_cnn":
        model = CustomCNN(num_classes=num_classes, dropout_rate=dropout_rate)
        head_names = ["classifier"]
        return model, head_names

    elif model_name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vgg16(weights=weights)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        head_names = ["classifier.6"]

    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        head_names = ["classifier.1"]

    elif model_name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        head_names = ["fc"]

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        head_names = ["classifier.1"]

    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )
        head_names = ["heads.head"]

    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")

    if mode == "feature_extraction":
        freeze_backbone(model, head_names)
    elif mode in ["fine_tuning", "scratch"]:
        for param in model.parameters():
            param.requires_grad = True

    return model, head_names


def get_model_target_layer(model_name: str, model: nn.Module) -> Optional[nn.Module]:
    """Retrieve the designated target layer for Grad-CAM."""
    model_name = model_name.lower()
    if model_name == "custom_cnn":
        return model.stage4[0].block[0]
    elif model_name == "vgg16":
        return model.features[28]
    elif model_name == "mobilenet_v2":
        return model.features[18][0]
    elif model_name == "resnet50":
        return model.layer4[2].conv3
    elif model_name == "efficientnet_b0":
        return model.features[8][0]
    elif model_name == "vit_b_16":
        return None  # ViT uses Attention Rollout instead of Grad-CAM
    return None
