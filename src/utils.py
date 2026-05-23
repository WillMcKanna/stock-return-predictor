# utils.py
# Purpose: creates helpers that can be shared across files.
# Reduces copy-pasting, allows for easy reproducibility

# Imports dependencies
import logging
import random
import numpy as np
import torch
import yaml

# Sets random seeds (random_seed 42 is standard). Allows for reproducibility.
# Need to call this function in any notebook that trains a model.
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes CUDA ops deterministic — slight performance cost, worth it for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# Cleans up output and gives named logger
# log = get_logger(__name__)
# Then, can use .debug, .info, .warning, .error to log progress
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

# Loads the project config from prepared file
# config = load_config() !! No args needed !!
# From here, can grab relevant info from the config
# Ex. start_date = config["data"]["start_date"]
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

# Returns # of trainable parameters in PyTorch model
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
