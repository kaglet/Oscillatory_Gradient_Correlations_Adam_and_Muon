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
    trainer: Trainer,
    model_cache: nn.Module, # as built class instance
    params_cache: Callable, # as getter method instance
    optims_cache: list[Partial[Optimizer]],
    schdlrs_cache: list[Partial[LRScheduler]]
):
    # print(store.has_enqueued())
    # Set matrix multiplication precision
    # TODO Fit this where it is more sensible, or make configurable, seems it should be in main loop at some torch.config level not composed into anything
    # torch.set_float32_matmul_precision("high")
    # logger = logging.getLogger(__name__)
    # logger.info("Logging from train and val input args", optims_cache[0], params_cache, model_cache, lit_module, datamodule)

    params = params_cache(model=model_cache)
    optims = [optim(params) for optim, params in zip(optims_cache, params)]
    # schdlrs = [sch(optimizer=optims[0]) for sch in schdlrs_cache]
    lit_module_built = lit_module(model=model_cache, params=params, optims=optims)
    trainer.fit(model=lit_module_built, datamodule=datamodule)
    
    return 1