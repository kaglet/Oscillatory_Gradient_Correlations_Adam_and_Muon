import logging
from typing import Callable, Literal, Optional, Type

import torch
import lightning as L
from lightning.pytorch.utilities import disable_possible_user_warnings
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import v2, ToTensor

__all__ = ["CIFAR10DataModule", "CIFAR100DataModule", "CIFARDataModule"]

class CIFARDataModule(L.LightningDataModule):
    # TODO" Unify the data directory from env see old template
    def __init__(self, train_transforms: Callable, val_transforms: Callable, test_transforms: Callable, dataset: Type[CIFAR10] | Type[CIFAR100] = CIFAR10, data_dir: str = "/data", batch_size: int = 2048, num_workers: int = 8) -> None:
        super().__init__()
        self.data_dir = data_dir
        # TODO: Resize images smaller and train on subset for efficiency and MVP.

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
            cifar_full = self.ds(self.data_dir, train=True, transform=self.train_transforms)
            self._cifar_train, self._cifar_val = random_split(
                cifar_full, [0.8, 0.2]
            )

        # Ensure test is never mixed up with train, val for any model, and is seen once and once only at the end
        if stage == "test":
            self._cifar_test = self.ds(self.data_dir, train=False, transform=self.test_transforms)

        if stage == "predict":
            self._cifar_predict = self.ds(self.data_dir, train=False, transform=self.test_transforms)

    # Define dataloaders below. Do not need to call ourselves, Trainer uses them. 
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._cifar_train, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._cifar_val, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._cifar_test, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=True)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self._cifar_predict, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=True)
    
class CIFAR10DataModule(CIFARDataModule):
    """ 
    Convenience class. This could be configured via hydra-zen. 
    This is in case different functionality is added not only for configuration runtime, for method overrides. 
    """
    def __init__(self, train_transforms: Callable, val_transforms: Callable, test_transforms: Callable, dataset: Type[CIFAR10] = CIFAR10, data_dir: str = "/data", batch_size: int = 2048, num_workers: int = 8) -> None:
        super().__init__(train_transforms, val_transforms, test_transforms, dataset, data_dir, batch_size, num_workers)

class CIFAR100DataModule(CIFARDataModule):
    """ 
    Convenience class. This could be configured via hydra-zen. 
    This is in case different functionality is added not only for configuration runtime, for method overrides. 
    """
    def __init__(self, train_transforms: Callable, val_transforms: Callable, test_transforms: Callable, dataset: Type[CIFAR100] = CIFAR100, data_dir: str = "/data", batch_size: int = 2048, num_workers: int = 8) -> None:
        super().__init__(train_transforms, val_transforms, test_transforms, dataset, data_dir, batch_size, num_workers)


if __name__ == "__main__":
    """
    Confirms smooth run of dataloader, native image sizes, with minimal transforms
    """
    # dmodule = CIFARDataModule() 
    # dmodule.prepare_data()
    # dmodule.setup("fit")
    # print(f"Train dataloader length: {len(dmodule.train_dataloader())}")

    # for i, (input, target) in enumerate(dmodule.train_dataloader()):
    #     print(f"Inputs of len {len(input)} targets of len {len(target)}")
    #     print(target)
    #     if i == 10:
    #         break
