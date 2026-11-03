"""Visualization tools for training curves, confusion matrices, and Pareto trade-off plots."""

from .plots import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_accuracy_vs_latency,
    plot_accuracy_vs_parameters,
    plot_cam_comparison,
)

__all__ = [
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_accuracy_vs_latency",
    "plot_accuracy_vs_parameters",
    "plot_cam_comparison",
]
