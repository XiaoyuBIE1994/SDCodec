#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2024 by Telecom-Paris
Authoried by Xiaoyu BIE (xiaoyu.bie@telecom-paris.fr)
License agreement in LICENSE.txt
"""

import os
import sys
import shutil
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

import torch

@hydra.main(version_base=None, config_path="config", config_name="default")
def main(cfg: DictConfig) -> None:

    from src.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()