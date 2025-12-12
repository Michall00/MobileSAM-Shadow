import logging
import random
import os
import torch
import numpy as np
from rich.logging import RichHandler

def setup_logger() -> None:
    """
    Configures the root logger to use RichHandler for console output
    while preserving existing file handlers (e.g., from Hydra).
    """
    root_logger = logging.getLogger()
    
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
        markup=True
    )
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    root_logger.addHandler(rich_handler)
    
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

def get_logger(name: str = __name__) -> logging.Logger:
    """Returns a logger instance with the specified name."""
    return logging.getLogger(name)

def set_seed(seed: int) -> None:
    """Sets the random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)