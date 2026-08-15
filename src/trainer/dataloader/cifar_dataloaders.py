# TODO: Create this dataloader

from typing import Callable, Literal, Optional, Type

import torch
import lightning as L
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import torch.nn as nn

__all__ = ["CIFAR10DataModule", "CIFAR100DataModule", "CIFARDataModule"]

class CIFARDataModule(L.LightningDataModule):
    # TODO" Unify the data directory from env see old template
    def __init__(self, train_transforms: Optional[Callable], val_transforms: Optional[Callable], test_transforms: Optional[Callable], dataset: Type[CIFAR10] | Type[CIFAR100] = CIFAR10, data_dir: str = "/data", batch_size: int = 2048, num_workers: int = 15) -> None:
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
        self.ds(self.data_dir, train=True, download=True)
        self.ds(self.data_dir, train=False, download=True)

    def setup(self, stage: str)  -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar_full = self.ds(self.data_dir, train=True, transform=self.train_transforms)
            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )

        # Ensure test is never mixed up with train, val for any model, and is seen once and once only at the end
        if stage == "test":
            self.cifar_test = self.ds(self.data_dir, train=False, transform=self.test_transforms)

        if stage == "predict":
            self.cifar_predict = self.ds(self.data_dir, train=False, transform=self.test_transforms)

    # Define dataloaders below. Do not need to call ourselves, Trainer uses them. 
    def train_dataloader(self) -> DataLoader:
        sampler = SubsetRandomSampler(self.train_indices.tolist())
        return DataLoader(self.cifar_train, batch_size=self.batch_size_per_device, num_workers=self.num_workers, sampler=sampler, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        sampler = SubsetRandomSampler(self.val_indices.tolist())
        return DataLoader(self.cifar_val, batch_size=self.batch_size_per_device, num_workers=self.num_workers, sampler=sampler, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        sampler = SubsetRandomSampler(self.test_indices.tolist())
        return DataLoader(self.cifar_test, batch_size=self.batch_size_per_device, num_workers=self.num_workers, sampler=sampler, persistent_workers=True)

    def predict_dataloader(self) -> DataLoader:
        sampler = SubsetRandomSampler(self.test_indices.tolist())
        return DataLoader(self.cifar_predict, batch_size=self.batch_size_per_device, num_workers=self.num_workers, sampler=sampler, persistent_workers=True)
    
class CIFAR10DataModule(CIFARDataModule):
    """ 
    Convenience class. This could be configured via hydra-zen. 
    This is in case different functionality is added not only for configuration runtime, for method overrides. 
    """
    def __init__(self, train_transforms: Optional[Callable], val_transforms: Optional[Callable], test_transforms: Optional[Callable], dataset: Type[CIFAR10] = CIFAR10, data_dir: str = "/data", batch_size: int = 2048, num_workers: int = 4) -> None:
        super().__init__(train_transforms, val_transforms, test_transforms, dataset, data_dir, batch_size, num_workers)

class CIFAR100DataModule(CIFARDataModule):
    """ 
    Convenience class. This could be configured via hydra-zen. 
    This is in case different functionality is added not only for configuration runtime, for method overrides. 
    """
    def __init__(self, train_transforms: Optional[Callable], val_transforms: Optional[Callable], test_transforms: Optional[Callable], dataset: Type[CIFAR100] = CIFAR100, data_dir: str = "/data", batch_size: int = 2048, num_workers: int = 4) -> None:
        super().__init__(train_transforms, val_transforms, test_transforms, dataset, data_dir, batch_size, num_workers)