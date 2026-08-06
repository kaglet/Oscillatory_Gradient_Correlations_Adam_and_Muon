import torch
import lightning as L
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import nn

class MNISTDataModule(L.LightningDataModule):
    # TODO: Unify the data directory from env see old template
    def __init__(self, train_transforms: nn.Module, val_transforms: nn.Module, test_transforms: nn.Module, data_dir: str = "/data", batch_size: int = 2048) -> None:
        super().__init__()
        self.data_dir = data_dir

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        self.batch_size_per_device = batch_size

    def prepare_data(self)  -> None:
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: str)  -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.train_transforms)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
            )

        # Ensure test is never mixed up with train, val for any model, and is seen once and once only at the end
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.test_transforms)

        if stage == "predict":
            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.test_transforms)

    # Define dataloaders below. Do not need to call ourselves, Trainer uses them. 
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_train, batch_size=self.batch_size_per_device)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_val, batch_size=self.batch_size_per_device)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_test, batch_size=self.batch_size_per_device)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.mnist_predict, batch_size=self.batch_size_per_device)