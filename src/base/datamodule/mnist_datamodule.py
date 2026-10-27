import os
from typing import Callable, Optional, Type

import torch
import lightning as L
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import nn
from tqdm import tqdm

__all__ = ["MNISTDataModule"]

class MNISTDataModule(L.LightningDataModule):
    # TODO: Unify the data directory from env see old template
    def __init__(self, dataset: Type[MNIST], train_transforms: Optional[Callable], val_transforms: Optional[Callable], test_transforms: Optional[Callable], data_dir: str = "./data", batch_size: int = 2048, num_workers: int = 0, persistent_workers:bool=False, download:bool=False) -> None:
        super().__init__()
        self.download = download
        self.data_dir = data_dir

        self._persistent_workers = persistent_workers

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        self.batch_size_per_device = batch_size
        self.num_workers = num_workers
        self.ds = dataset

    def prepare_data(self)  -> None:
        # download
        self.ds(self.data_dir, train=True, download=True)
        self.ds(self.data_dir, train=False, download=True)

    def setup(self, stage: str)  -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.train_transforms)
            self._train, self._val = random_split(
                mnist_full, [0.8, 0.2]
            )

        # Ensure test is never mixed up with train, val for any model, and is seen once and once only at the end
        if stage == "test":
            self._test = MNIST(self.data_dir, train=False, transform=self.test_transforms)

        if stage == "predict":
            self._predict = MNIST(self.data_dir, train=False, transform=self.test_transforms)

    # Define dataloaders below. Do not need to call ourselves, Trainer uses them. 
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=self._persistent_workers, drop_last=True, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=self._persistent_workers, drop_last=True, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=self._persistent_workers, drop_last=True, shuffle=True)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self._predict, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=self._persistent_workers, drop_last=True, shuffle=True) 

