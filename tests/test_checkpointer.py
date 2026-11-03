import os
import tempfile
import torch
import torch.nn as nn
from cv_benchmark.training.checkpointer import Checkpointer


def test_checkpoint_save_and_load():
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = Checkpointer(checkpoint_dir=tmpdir)
        model = nn.Linear(10, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        state = {
            "epoch": 3,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": 0.42,
        }

        saved_path = checkpointer.save_checkpoint(state, is_best=True)
        assert os.path.exists(saved_path)
        assert os.path.exists(os.path.join(tmpdir, "model_best.pt"))

        # Load into fresh model
        new_model = nn.Linear(10, 2)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        loaded = checkpointer.load_checkpoint(saved_path, new_model, new_optimizer)

        assert loaded["epoch"] == 3
        assert loaded["best_val_loss"] == 0.42
