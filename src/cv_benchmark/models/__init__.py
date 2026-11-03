"""Model definitions and factory for CV Benchmark architectures."""

from .custom_cnn import CustomCNN
from .factory import create_model, get_model_target_layer

__all__ = ["CustomCNN", "create_model", "get_model_target_layer"]
