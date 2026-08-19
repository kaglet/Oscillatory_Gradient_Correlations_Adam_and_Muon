import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer, LightningDataModule, LightningModule

def train_and_val(
    lit_module: LightningModule,
    datamodule: LightningDataModule,
    trainer: Trainer,
):
    # Set matrix multiplication precision
    # TODO Fit this where it is more sensible, or make configurable, seems it should be in main loop at some torch.config level not composed into anything
    # torch.set_float32_matmul_precision("high")
    trainer.fit(model=lit_module, datamodule=datamodule)
    # TODO: When you run test evaluate best AND last for Muon to see if degradation is a notably suffered effect
    trainer.test(model=lit_module, datamodule=datamodule)
    
    return 1