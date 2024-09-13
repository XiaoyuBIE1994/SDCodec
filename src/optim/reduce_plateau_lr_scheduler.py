

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class ReducePlateauLRScheduler(_LRScheduler):
    """Cosine LR scheduler.
    Add warmup steps for torch.optim.lr_scheduler.ReduceLROnPlateau

    Args:
        optimizer (Optimizer): Torch optimizer.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of steps.
        lr_min_ratio (float): Minimum learning rate.
        cycle_length (float): Cycle length.
    """

    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_steps: int,
                 lr_min_ratio: float = 0.0, cycle_length: float = 1.0):
        # TODO
        pass