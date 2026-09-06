import os
from typing import Callable, Optional

import torch
import lightning as L
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import nn
from tqdm import tqdm

__all__ = ["SimpleMNISTDataModule", "MNISTDataModule"]

class MNISTDataModule(L.LightningDataModule):
    # TODO: Unify the data directory from env see old template
    def __init__(self, train_transforms: Optional[Callable], val_transforms: Optional[Callable], test_transforms: Optional[Callable], data_dir: str = "/data", batch_size: int = 2048, num_workers: int = 0) -> None:
        super().__init__()
        self.data_dir = data_dir

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        self.batch_size_per_device = batch_size
        self.num_workers = num_workers

    def prepare_data(self)  -> None:
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str)  -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.train_transforms)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [0.8, 0.2]
            )

        # Ensure test is never mixed up with train, val for any model, and is seen once and once only at the end
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.test_transforms)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.test_transforms)

    # Define dataloaders below. Do not need to call ourselves, Trainer uses them. 
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.batch_size_per_device, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.batch_size_per_device, num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.batch_size_per_device, num_workers=self.num_workers)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_predict, batch_size=self.batch_size_per_device, num_workers=self.num_workers) 

    def get_train_data_stats(self):
        dloader = self.train_dataloader()
        pix_sum = torch.zeros(size=(3,), dtype=torch.float32)
        pix_sum_sq = torch.zeros_like(pix_sum)
        img_dim = self.mnist_train[0]
        img_pix_count = img_dim[1] * img_dim[2]
        num_img = len(self.mnist_train)
        total_pix_count = num_img * img_pix_count
        for input, target in tqdm(dloader):
            pix_sum += input.sum(axis=[1,2])
            pix_sum_sq += input.square().sum(axis=[1,2])

        total_mean = pix_sum / total_pix_count
        total_var = (pix_sum_sq / total_pix_count) - (total_mean.square())
        total_std = torch.sqrt(total_var)

        return total_mean, total_std

class SimpleMNISTDataModule(L.LightningDataModule):

  def setup(self, stage):
    # transforms for images
    transform=transforms.Compose([transforms.ToTensor(), 
                                  transforms.Normalize((0.1307,), (0.3081,))])
      
    # prepare transforms standard to MNIST
    self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    self.mnist_train, self.mnist_val = random_split(
                    self.mnist_train, [0.8, 0.2]
                )
    self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)

  def train_dataloader(self):
    return DataLoader(self.mnist_train, batch_size=64, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.mnist_val, batch_size=64, shuffle=False)

  def test_dataloader(self):
    return DataLoader(self.mnist_val, batch_size=64, shuffle=False)