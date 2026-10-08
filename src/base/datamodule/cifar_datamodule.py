import logging
from typing import Any, Callable, Literal, Optional, Type

import torch
import lightning as L
from lightning.pytorch.utilities import disable_possible_user_warnings
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import ToTensor

from src.base.datamodule.components.convert_ax import convert_img_axes
from src.base.datamodule.components.img_viz import viz_img

__all__ = ["CIFAR10DataModule", "CIFAR100DataModule", "CIFARDataModule"]

class CIFARDataModule(L.LightningDataModule):
    # TODO" Unify the data directory from env see old template
    def __init__(self, train_transforms: transforms.Compose, val_transforms: transforms.Compose, test_transforms: transforms.Compose, dataset: Type[CIFAR10] | Type[CIFAR100] = CIFAR10, data_dir: str = "/data", batch_size: int = 2048, num_workers: int = 8) -> None:
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
            self._train, self._val = random_split(
                cifar_full, [0.8, 0.2]
            )

        # Ensure test is never mixed up with train, val for any model, and is seen once and once only at the end
        if stage == "test":
            self._test = self.ds(self.data_dir, train=False, transform=self.test_transforms)

        if stage == "predict":
            self._predict = self.ds(self.data_dir, train=False, transform=self.test_transforms)

    # Define dataloaders below. Do not need to call ourselves, Trainer uses them. 
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=False, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=False, drop_last=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=False, drop_last=True)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self._predict, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=False, drop_last=True)
    
class CIFAR10DataModule(CIFARDataModule):
    """ 
    Convenience class. This could be configured via hydra-zen. 
    This is in case different functionality is added not only for configuration runtime, for method overrides. 
    """
    def __init__(*args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

class CIFAR100DataModule(CIFARDataModule):
    """ 
    Convenience class. This could be configured via hydra-zen. 
    This is in case different functionality is added not only for configuration runtime, for method overrides. 
    """
    def __init__(*args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

if __name__ == "__main__":
    """
    Confirms smooth run of CIFAR10 dataloader, native image sizes, with minimal transforms
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # ImageNet standard norm
        std=[0.229, 0.224, 0.225]
    )
    
    base_transforms = transforms.Compose([
        ToTensor(),
        normalize
    ])

    dmodule = CIFARDataModule(
        num_workers=1, 
        train_transforms=base_transforms, 
        val_transforms=base_transforms, 
        test_transforms=base_transforms
    ) 
    dmodule.prepare_data()
    dmodule.setup("fit")
    print(f"Train dataloader length: {len(dmodule.train_dataloader())}")

    i = 0
    for inputs, targets in dmodule.train_dataloader():
        img = convert_img_axes(inputs[0])
        viz_img(torch.as_tensor(img), label=targets[0])
        i += 1
        if i == 10: 
            break

# Todo: Perform this as experiments via config, use of ratio and other settings, move ImageNet norm to other datasets not this one. 