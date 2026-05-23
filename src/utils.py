"""
utils.py

Shared helpers: reproducibility, logging, config loading.
Import from here rather than repeating boilerplate across files.

Usage:
    from src.utils import set_seed, load_config, get_logger
"""

import logging
import random

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
    Call this at the top of train.py and any notebook that trains a model.

    Args:
        seed : integer seed (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes CUDA ops deterministic — slight performance cost, worth it for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a named logger with a consistent format.
    Prevents duplicate handlers if called multiple times.

    Usage:
        log = get_logger(__name__)
        log.info("Training started.")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_config(path: str = "config.yaml") -> dict:
    """
    Loads the project config from a YAML file.

    Usage:
        config = load_config()
        start_date = config["data"]["start_date"]
    """
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


def count_parameters(model: torch.nn.Module) -> int:
    """
    Returns the number of trainable parameters in a PyTorch model.
    Useful for comparing model complexity in your write-up.

    Usage:
        print(f"Parameters: {count_parameters(model):,}")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
