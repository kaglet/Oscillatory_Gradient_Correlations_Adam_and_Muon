import inspect
import os
from pathlib import Path
from typing import Any, Callable

import logging
import hydra
import torch
import torch.nn as nn
from torch.optim import Optimizer  
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, LightningDataModule, LightningModule
from hydra_zen.typing import Partial
from hydra_zen import ZenStore, store
from tqdm import tqdm
import lightning.pytorch as pl

from src.base.configs.experiment_data import Experiment_Data
__all__ = ["train_and_val"]

def train_and_val(
    lit_module: Partial[LightningModule],
    datamodule: LightningDataModule,
    trainer: Partial[Trainer],
    model_cache: nn.Module, # as built class instance
    params_cache: Callable, # as getter method instance
    optims_cache: list[Partial[Optimizer]],
    schdlr_cache: Partial[LRScheduler], 
    logger_cache: Any,
    chkpt_cache: Any,
    run_idx: int,
    seed: int
):
    pl.seed_everything(seed)
    torch.set_float32_matmul_precision(precision="high")
    print(f"Executing run {run_idx}") 
    params = params_cache(model=model_cache)
    optims = [optim(params) for optim, params in zip(optims_cache, params)]
    schdlrs = [schdlr_cache(optimizer=optims[0]) for _ in optims]
    lit_module_built = lit_module(model=model_cache, params=params, optims=optims, schedulers=schdlrs)

    path_all = Path(os.getcwd(), f"run_{run_idx}") # 
    trainer_built = trainer(default_root_dir=path_all)
    trainer_built.fit(model=lit_module_built, datamodule=datamodule)
    
    return 1