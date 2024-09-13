"""
Scheduler modified from AudioCraft project https://github.com/facebookresearch/audiocraft/tree/main
"""

# flake8: noqa
from .cosine_lr_scheduler import CosineLRScheduler
from .exponential_lr_scheduler import ExponentialLRScheduler
from .inverse_sqrt_lr_scheduler import InverseSquareRootLRScheduler
from .linear_warmup_lr_scheduler import LinearWarmupLRScheduler
from .polynomial_decay_lr_scheduler import PolynomialDecayLRScheduler
from .reduce_plateau_lr_scheduler import ReducePlateauLRScheduler