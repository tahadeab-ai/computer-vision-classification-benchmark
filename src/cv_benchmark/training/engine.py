from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def calculate_topk_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """Calculate Top-1 and Top-5 accuracy percentages."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f"top_{k}_acc"] = float(correct_k.mul_(100.0 / batch_size).item())
        return res


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    mixed_precision: bool = True,
    clip_grad_norm: float = 1.0,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Execute one training epoch with mixed-precision support and gradient clipping."""
    model.train()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    use_cuda = device.type == "cuda"
    amp_enabled = mixed_precision and use_cuda

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for step, (images, targets) in enumerate(pbar):
        if max_steps is not None and step >= max_steps:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
            optimizer.step()

        accs = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
        total_loss += loss.item() * batch_size
        total_top1 += (accs["top_1_acc"] / 100.0) * batch_size
        total_top5 += (accs["top_5_acc"] / 100.0) * batch_size
        total_samples += batch_size

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "top1": f"{accs['top_1_acc']:.2f}%"})

    avg_loss = total_loss / max(total_samples, 1)
    avg_top1 = (total_top1 / max(total_samples, 1)) * 100.0
    avg_top5 = (total_top5 / max(total_samples, 1)) * 100.0

    return {
        "train_loss": avg_loss,
        "train_top1": avg_top1,
        "train_top5": avg_top5,
    }


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mixed_precision: bool = True,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    """Execute one validation epoch evaluating loss and Top-1/Top-5 accuracy."""
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    use_cuda = device.type == "cuda"
    amp_enabled = mixed_precision and use_cuda

    pbar = tqdm(dataloader, desc="Validation", leave=False)
    for step, (images, targets) in enumerate(pbar):
        if max_steps is not None and step >= max_steps:
            break

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = images.size(0)

        with torch.amp.autocast(device_type="cuda" if use_cuda else "cpu", enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)

        accs = calculate_topk_accuracy(outputs, targets, topk=(1, 5))
        total_loss += loss.item() * batch_size
        total_top1 += (accs["top_1_acc"] / 100.0) * batch_size
        total_top5 += (accs["top_5_acc"] / 100.0) * batch_size
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_top1 = (total_top1 / max(total_samples, 1)) * 100.0
    avg_top5 = (total_top5 / max(total_samples, 1)) * 100.0

    return {
        "val_loss": avg_loss,
        "val_top1": avg_top1,
        "val_top5": avg_top5,
    }
