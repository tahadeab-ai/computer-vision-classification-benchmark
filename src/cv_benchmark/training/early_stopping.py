class EarlyStopping:
    """Tracks validation metrics and signals when to stop training early."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "min",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        """Evaluate current metric score. Returns True if improvement was observed."""
        if self.mode == "min":
            improvement = (self.best_score - current_score) > self.min_delta
        else:
            improvement = (current_score - self.best_score) > self.min_delta

        if improvement:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
