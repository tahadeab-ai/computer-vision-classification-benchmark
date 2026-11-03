import os
from typing import Any, Dict
import yaml


def load_yaml(file_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file safely."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = dict1.copy()
    for key, val in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(
    dataset_cfg_path: str = "configs/dataset.yaml",
    models_cfg_path: str = "configs/models.yaml",
    training_cfg_path: str = "configs/training.yaml",
) -> Dict[str, Any]:
    """Load and merge all configuration files."""
    dataset_cfg = load_yaml(dataset_cfg_path)
    models_cfg = load_yaml(models_cfg_path)
    training_cfg = load_yaml(training_cfg_path)

    merged = {}
    merged.update(dataset_cfg)
    merged.update(models_cfg)
    merged.update(training_cfg)
    return merged
