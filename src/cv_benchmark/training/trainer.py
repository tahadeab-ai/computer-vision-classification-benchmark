import os
import json
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .engine import train_one_epoch, validate_one_epoch
from .checkpointer import Checkpointer
from .early_stopping import EarlyStopping

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class Trainer:
    """End-to-end training orchestrator supporting AMP, early stopping, and recovery."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        experiment_name: str = "cifar100_experiment",
        output_dir: str = "./reports",
        checkpoint_dir: str = "./checkpoints",
        tensorboard_dir: str = "./runs",
        mixed_precision: bool = True,
        clip_grad_norm: float = 1.0,
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 0.001,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.mixed_precision = mixed_precision
        self.clip_grad_norm = clip_grad_norm

        self.model.to(self.device)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(mixed_precision and self.device.type == "cuda"))

        self.checkpointer = Checkpointer(checkpoint_dir=os.path.join(checkpoint_dir, experiment_name))
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode="min",
        )

        self.writer = None
        if SummaryWriter is not None and tensorboard_dir:
            log_dir = os.path.join(tensorboard_dir, experiment_name)
            self.writer = SummaryWriter(log_dir=log_dir)

        self.history: Dict[str, list] = {
            "epoch": [],
            "train_loss": [],
            "train_top1": [],
            "train_top5": [],
            "val_loss": [],
            "val_top1": [],
            "val_top5": [],
            "lr": [],
        }
        self.best_val_loss = float("inf")
        self.start_epoch = 1

    def fit(self, epochs: int, max_steps_per_epoch: Optional[int] = None) -> Dict[str, list]:
        """Execute the training loop over specified epochs."""
        print(f"\n[Trainer] Starting training for {self.experiment_name} on device: {self.device}")
        os.makedirs(self.output_dir, exist_ok=True)

        for epoch in range(self.start_epoch, epochs + 1):
            current_lr = self.optimizer.param_groups[0]["lr"]

            train_metrics = train_one_epoch(
                model=self.model,
                dataloader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
                scaler=self.scaler,
                mixed_precision=self.mixed_precision,
                clip_grad_norm=self.clip_grad_norm,
                max_steps=max_steps_per_epoch,
            )

            val_metrics = validate_one_epoch(
                model=self.model,
                dataloader=self.val_loader,
                criterion=self.criterion,
                device=self.device,
                mixed_precision=self.mixed_precision,
                max_steps=max_steps_per_epoch,
            )

            if self.scheduler is not None:
                self.scheduler.step()

            # Record metrics
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["train_top1"].append(train_metrics["train_top1"])
            self.history["train_top5"].append(train_metrics["train_top5"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_top1"].append(val_metrics["val_top1"])
            self.history["val_top5"].append(val_metrics["val_top5"])
            self.history["lr"].append(current_lr)

            # Tensorboard logging
            if self.writer:
                self.writer.add_scalar("Loss/train", train_metrics["train_loss"], epoch)
                self.writer.add_scalar("Loss/val", val_metrics["val_loss"], epoch)
                self.writer.add_scalar("Accuracy/train_top1", train_metrics["train_top1"], epoch)
                self.writer.add_scalar("Accuracy/val_top1", val_metrics["val_top1"], epoch)
                self.writer.add_scalar("LearningRate", current_lr, epoch)

            is_best = val_metrics["val_loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["val_loss"]

            # Save checkpoint state atomically
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            }
            self.checkpointer.save_checkpoint(
                state=checkpoint_state,
                is_best=is_best,
                filename="checkpoint_latest.pt",
                best_filename="model_best.pt",
            )

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train Loss: {train_metrics['train_loss']:.4f} (Top1: {train_metrics['train_top1']:.2f}%) | "
                f"Val Loss: {val_metrics['val_loss']:.4f} (Top1: {val_metrics['val_top1']:.2f}%)"
            )

            # Check early stopping
            improved = self.early_stopping(val_metrics["val_loss"])
            if self.early_stopping.early_stop:
                print(f"[Trainer] Early stopping triggered at epoch {epoch}.")
                break

        if self.writer:
            self.writer.close()

        # Save training history JSON
        history_path = os.path.join(self.output_dir, f"{self.experiment_name}_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

        return self.history
