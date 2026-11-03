import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from cv_benchmark.training.engine import calculate_topk_accuracy, train_one_epoch, validate_one_epoch


def test_topk_accuracy():
    # 2 samples, 5 classes
    logits = torch.tensor([[10.0, 5.0, 1.0, 0.0, 0.0], [1.0, 2.0, 8.0, 0.0, 0.0]])
    targets = torch.tensor([0, 2])
    res = calculate_topk_accuracy(logits, targets, topk=(1, 5))
    assert res["top_1_acc"] == 100.0
    assert res["top_5_acc"] == 100.0


def test_train_and_validate_one_epoch():
    # Simple linear model and dummy dataloader
    model = nn.Linear(10, 5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    device = torch.device("cpu")

    x = torch.randn(16, 10)
    y = torch.randint(0, 5, (16,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)

    train_res = train_one_epoch(
        model=model,
        dataloader=loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        mixed_precision=False,
    )
    assert "train_loss" in train_res
    assert "train_top1" in train_res

    val_res = validate_one_epoch(
        model=model,
        dataloader=loader,
        criterion=criterion,
        device=device,
        mixed_precision=False,
    )
    assert "val_loss" in val_res
    assert "val_top1" in val_res
