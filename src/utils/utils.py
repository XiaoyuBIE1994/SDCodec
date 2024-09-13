import os
import sys
import math
import json
from typing import List, Union

def warm_up(step, warmup_steps):
    if step < warmup_steps:
        warmup_ratio = step / warmup_steps
    else:
        warmup_ratio = 1
    return warmup_ratio


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.hist = []
        self.reset()

    def reset(self):
        self.val = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        self.avg += (self.val - self.avg) * n / self.count

    def save_log(self):
        self.hist.append(self.avg)
        self.reset()


class TrainMonitor(object):
    """Record training"""

    def __init__(self, nb_step=1, best_eval=math.inf, best_step=1, early_stop=0):
        self.nb_step = nb_step
        self.best_eval = best_eval
        self.best_step = best_step
        self.early_stop = early_stop


    def state_dict(self):
        sd = {'nb_step': self.nb_step,
              'best_eval': self.best_eval,
              'best_step': self.best_step,
              'early_stop': self.early_stop,
            }
        return sd

    
    def load_state_dict(self, state_dict: dict):
        self.nb_step = state_dict['nb_step']
        self.best_eval = state_dict['best_eval']
        self.best_step = state_dict['best_step']
        self.early_stop = state_dict['early_stop']