# TODO: Create this dataloader

from typing import Callable, Literal, Optional, Type

import torch
import lightning as L
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CelebA
from torchvision import transforms
import torch.nn as nn

__all__ = ["CelebADataModule"]

class CelebADataModule(L.LightningDataModule):
    # TODO" Unify the data directory from env see old template
    def __init__(self, train_transforms: Optional[Callable], val_transforms: Optional[Callable], test_transforms: Optional[Callable], dataset: Type[CelebA] = CelebA, data_dir: str = "./data", batch_size: int = 2048, num_workers: int = 15) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        self.batch_size_per_device = batch_size
        self.num_workers = num_workers
        self.ds = dataset

        self.train_indices = torch.randint(0, 5000, (100,))
        self.val_indices = torch.randint(5000, 6000, (20,))
        self.test_indices = torch.randint(9000, 10000, (30,))

    def prepare_data(self)  -> None:
        # download
        self.ds(self.data_dir, split="all", download=True, target_type="attr")

    def setup(self, stage: str)  -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            train = self.ds(self.data_dir, split="train", transform=self.train_transforms)
            val = self.ds(self.data_dir, split="valid", transform=self.train_transforms)
            self.celeb_a_train, self.celeb_a_val = train, val

        # Ensure test is never mixed up with train, val for any model, and is seen once and once only at the end
        if stage == "test":
            self.celeb_a_test = self.ds(self.data_dir, split="test", transform=self.test_transforms)

        if stage == "predict":
            self.celeb_a_predict = self.ds(self.data_dir, split="test", transform=self.test_transforms)

    # Define dataloaders below. Do not need to call ourselves, Trainer uses them. 
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.celeb_a_train, batch_size=self.batch_size_per_device, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.celeb_a_val, batch_size=self.batch_size_per_device, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.celeb_a_test, batch_size=self.batch_size_per_device, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.celeb_a_predict, batch_size=self.batch_size_per_device, num_workers=self.num_workers)
    