import os
from pathlib import Path
from typing import Any, Callable

import logging
import torch
import torch.nn as nn
from torch.optim import Optimizer  
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, LightningDataModule, LightningModule
from hydra_zen.typing import Partial
from hydra_zen import ZenStore, store
__all__ = ["train_and_val"]

def train_and_val(
    lit_module: Partial[LightningModule],
    datamodule: LightningDataModule,
    trainer: Partial[Trainer],
    model_cache: nn.Module, # as built class instance
    params_cache: Callable, # as getter method instance
    optims_cache: list[Partial[Optimizer]],
    schdlrs_cache: list[Partial[LRScheduler]],
    experiment_data: Any,
    logger_cache: Any,
    chkpt_cache: Any
):
    params = params_cache(model=model_cache)
    optims = [optim(params) for optim, params in zip(optims_cache, params)]
    # schdlrs = [sch(optimizer=optims[0]) for sch in schdlrs_cache]
    lit_module_built = lit_module(model=model_cache, params=params, optims=optims)

    for i in range(2):
        path_all = Path(os.path.join(experiment_data.log_path, f"run_{i}"))
        os.makedirs(path_all, exist_ok=True)
        loggers_built = [lg(save_dir=path_all) for lg in logger_cache]
        trainer_built = trainer(default_root_dir=path_all, logger=loggers_built)
        trainer_built.fit(model=lit_module_built, datamodule=datamodule)
    
    return 1