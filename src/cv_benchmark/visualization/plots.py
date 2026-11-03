import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_curves(history: Dict[str, list], save_path: str, title: str = "Training & Validation Metrics") -> None:
    """Plot Loss and Top-1 Accuracy curves across training epochs."""
    epochs = history.get("epoch", list(range(1, len(history["train_loss"]) + 1)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    ax1.plot(epochs, history["train_loss"], label="Train Loss", marker="o", color="#2b5c8f")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", marker="s", color="#d95f02")
    ax1.set_title(f"{title} - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # Accuracy curve
    ax2.plot(epochs, history["train_top1"], label="Train Top-1 Acc", marker="o", color="#2b5c8f")
    ax2.plot(epochs, history["val_top1"], label="Val Top-1 Acc", marker="s", color="#d95f02")
    ax2.set_title(f"{title} - Top-1 Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]],
    save_path: str,
    top_n_classes: int = 20,
) -> None:
    """Plot an annotated confusion matrix for the most active / confused classes."""
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    # Subsample if 100 classes to keep visualization readable
    if cm.shape[0] > top_n_classes:
        per_class_samples = cm.sum(axis=1)
        top_indices = np.argsort(per_class_samples)[-top_n_classes:]
        cm_sub = cm[np.ix_(top_indices, top_indices)]
        labels_sub = [class_names[i] for i in top_indices]
    else:
        cm_sub = cm
        labels_sub = class_names

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_sub,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_sub,
        yticklabels=labels_sub,
    )
    plt.title(f"Confusion Matrix (Top {len(labels_sub)} Classes)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_accuracy_vs_latency(
    benchmark_data: List[Dict],
    save_path: str,
) -> None:
    """Plot accuracy vs latency Pareto trade-off scatter plot."""
    plt.figure(figsize=(10, 6))

    for item in benchmark_data:
        name = item["model_name"]
        latency = item["latency_mean_ms"]
        acc = item["top1_accuracy"]
        params = item.get("total_parameters_m", 10.0)

        # Bubble size proportional to parameter count
        size = max(50, min(1200, params * 12))
        plt.scatter(latency, acc, s=size, alpha=0.7, label=name)
        plt.annotate(
            f"{name}\n({acc:.1f}%, {latency:.1f}ms)",
            (latency, acc),
            textcoords="offset points",
            xytext=(10, 5),
            ha="left",
            fontsize=9,
            weight="bold",
        )

    plt.title("Accuracy vs. Latency Trade-Off (CIFAR-100 Benchmark)")
    plt.xlabel("Mean Inference Latency (ms)")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_accuracy_vs_parameters(
    benchmark_data: List[Dict],
    save_path: str,
) -> None:
    """Plot accuracy vs parameter count trade-off scatter plot."""
    plt.figure(figsize=(10, 6))

    for item in benchmark_data:
        name = item["model_name"]
        params = item["total_parameters_m"]
        acc = item["top1_accuracy"]

        plt.scatter(params, acc, s=150, alpha=0.8, label=name)
        plt.annotate(
            f"{name} ({acc:.1f}%)",
            (params, acc),
            textcoords="offset points",
            xytext=(8, 4),
            ha="left",
            fontsize=9,
        )

    plt.title("Accuracy vs. Model Capacity (Parameters)")
    plt.xlabel("Total Parameters (Millions)")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_cam_comparison(
    image: np.ndarray,
    cam: np.ndarray,
    overlay: np.ndarray,
    class_label: str,
    save_path: str,
) -> None:
    """Plot Input, Grad-CAM Heatmap, and Overlay side-by-side."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(image)
    axes[0].set_title(f"Original: {class_label}")
    axes[0].axis("off")

    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
