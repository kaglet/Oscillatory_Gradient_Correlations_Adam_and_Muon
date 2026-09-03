from typing import Callable, Literal, Optional, Type

import torch
import lightning as L
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import v2, ToTensor

__all__ = ["CIFAR10DataModule", "CIFAR100DataModule", "CIFARDataModule"]

class CIFARDataModule(L.LightningDataModule):
    # TODO" Unify the data directory from env see old template
    def __init__(self, train_transforms: Optional[Callable] | None = None, val_transforms: Optional[Callable] | None = None, test_transforms: Optional[Callable] | None = None, dataset: Type[CIFAR10] | Type[CIFAR100] = CIFAR10, data_dir: str = "/data", batch_size: int = 2048, num_workers: int = 15) -> None:
        super().__init__()
        self.data_dir = data_dir
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet standard norm
            std=[0.229, 0.224, 0.225]
        )
        default_transforms = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            ToTensor(),
            normalize
        ])

        self.train_transforms = train_transforms if train_transforms is not None else default_transforms
        self.val_transforms = val_transforms if val_transforms is not None else default_transforms
        self.test_transforms = test_transforms if test_transforms is not None else default_transforms

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
        return DataLoader(self._cifar_train, batch_size=self.batch_size_per_device, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._cifar_val, batch_size=self.batch_size_per_device, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._cifar_test, batch_size=self.batch_size_per_device, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self._cifar_predict, batch_size=self.batch_size_per_device, num_workers=self.num_workers)
    
class CIFAR10DataModule(CIFARDataModule):
    """ 
    Convenience class. This could be configured via hydra-zen. 
    This is in case different functionality is added not only for configuration runtime, for method overrides. 
    """
    def __init__(self, train_transforms: Optional[Callable] | None = None, val_transforms: Optional[Callable] | None = None, test_transforms: Optional[Callable] | None = None, dataset: Type[CIFAR10] = CIFAR10, data_dir: str = "/data", batch_size: int = 2048, num_workers: int = 0) -> None:
        super().__init__(train_transforms, val_transforms, test_transforms, dataset, data_dir, batch_size, num_workers)

class CIFAR100DataModule(CIFARDataModule):
    """ 
    Convenience class. This could be configured via hydra-zen. 
    This is in case different functionality is added not only for configuration runtime, for method overrides. 
    """
    def __init__(self, train_transforms: Optional[Callable] | None = None, val_transforms: Optional[Callable] | None = None, test_transforms: Optional[Callable] | None = None, dataset: Type[CIFAR100] = CIFAR100, data_dir: str = "/data", batch_size: int = 2048, num_workers: int = 0) -> None:
        super().__init__(train_transforms, val_transforms, test_transforms, dataset, data_dir, batch_size, num_workers)