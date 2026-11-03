import os
import shutil
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class Checkpointer:
    """Manages atomic checkpoint saving, recovery, and version preservation."""

    def __init__(self, checkpoint_dir: str = "./checkpoints") -> None:
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        is_best: bool,
        filename: str = "checkpoint_latest.pt",
        best_filename: str = "model_best.pt",
    ) -> str:
        """Atomically saves checkpoint and updates best model without overwriting destructively."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        tmp_filepath = filepath + ".tmp"

        torch.save(state, tmp_filepath)
        if os.path.exists(filepath):
            # Preserve previous checkpoint as backup
            backup_filepath = filepath + ".bak"
            shutil.copyfile(filepath, backup_filepath)
        os.replace(tmp_filepath, filepath)

        if is_best:
            best_filepath = os.path.join(self.checkpoint_dir, best_filename)
            shutil.copyfile(filepath, best_filepath)

        return filepath

    def load_checkpoint(
        self,
        filepath: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint dictionary and restore weights, optimizer, and scheduler states."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

        checkpoint = torch.load(filepath, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint
