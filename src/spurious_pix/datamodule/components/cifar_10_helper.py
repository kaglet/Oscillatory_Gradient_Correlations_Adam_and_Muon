from pathlib import Path
from typing import Any, Callable, Optional
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
import torch 
from PIL import Image
import numpy as np

__all__ = ["MyMNIST"]

class MyMNIST(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = data 
        self.targets = targets 

    def __getitem__(self, index) -> tuple[Any, Any]:
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.targets)