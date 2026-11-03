import numpy as np
from cv_benchmark.evaluation.metrics import compute_classification_metrics


def test_compute_classification_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1])
    
    # 3 classes: 4 correct out of 6 -> 66.67%
    metrics = compute_classification_metrics(y_true, y_pred, num_classes=3)

    assert "top1_accuracy" in metrics
    assert metrics["top1_accuracy"] == 66.67
    assert "f1_macro" in metrics
    assert "confusion_matrix" in metrics
    assert len(metrics["per_class_accuracy"]) == 3
