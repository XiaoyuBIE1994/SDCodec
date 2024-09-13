# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class ExponentialLRScheduler(_LRScheduler):
    """Exponential LR scheduler.

    Args:
        optimizer (Optimizer): Torch optimizer.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of steps.
        lr_min_ratio (float): Minimum learning rate.
        cycle_length (float): Cycle length.
    """
    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_steps: int,
                 lr_min_ratio: float = 0.0, gamma: float = 0.99999):
        self.warmup_steps = warmup_steps
        assert self.warmup_steps >= 0
        self.total_steps = total_steps
        assert self.total_steps >= warmup_steps
        self.lr_min_ratio = lr_min_ratio
        self.cumprod = 1
        self.gamma = gamma
        super().__init__(optimizer)

    def _get_sched_lr(self, lr: float, step: int):
        if step < self.warmup_steps:
            lr_ratio = step / self.warmup_steps
            lr = lr_ratio * lr
        elif step <= self.total_steps:
            self.cumprod *= self.gamma
            lr_ratio = self.cumprod
            lr = lr_ratio * lr
        else:
            lr_ratio = self.lr_min_ratio
            lr = lr_ratio * lr
        return lr

    def get_lr(self):
        return [self._get_sched_lr(lr, self.last_epoch) for lr in self.base_lrs]