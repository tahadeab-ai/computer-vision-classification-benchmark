from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: Optional[np.ndarray] = None,
    num_classes: int = 100,
) -> Dict[str, Any]:
    """Compute comprehensive classification evaluation metrics."""
    top1 = float(accuracy_score(y_true, y_pred) * 100.0)

    top5 = 0.0
    if y_probs is not None and y_probs.shape[1] >= 5:
        all_classes = np.arange(num_classes)
        top5 = float(top_k_accuracy_score(y_true, y_probs, k=5, labels=all_classes) * 100.0)

    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))

    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    # Per-class accuracy
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc)

    return {
        "top1_accuracy": round(top1, 2),
        "top5_accuracy": round(top5, 2),
        "precision_macro": round(precision_macro, 4),
        "precision_weighted": round(precision_weighted, 4),
        "recall_macro": round(recall_macro, 4),
        "recall_weighted": round(recall_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_accuracy": [round(float(x * 100.0), 2) for x in per_class_acc],
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 100,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a trained model on the test dataset and return metric dictionary."""
    model.eval()
    model.to(device)

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []

    for step, (images, targets) in enumerate(test_loader):
        if max_steps is not None and step >= max_steps:
            break

        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

        all_preds.extend(preds.tolist())
        all_targets.extend(targets.tolist())
        all_probs.append(probs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_probs = np.vstack(all_probs) if all_probs else None

    return compute_classification_metrics(y_true, y_pred, y_probs=y_probs, num_classes=num_classes)
