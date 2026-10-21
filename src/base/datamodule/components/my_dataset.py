import torch
from torch.utils.data import Dataset

from PIL import Image

from torchvision.datasets import CIFAR10, CIFAR100

import math
from typing import Any, Optional, Tuple
import logging
import skimage
import numpy as np
import os
from glob import glob
import torch
from torchvision.transforms import Resize
import torchvision
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from torchvision import transforms
from torchvision.transforms import Compose

from src.base.datamodule.components.convert_ax import convert_img_axes

logger = logging.getLogger(__name__)

class MyDataset(Dataset):
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, transforms:Compose, target_transforms: Optional[Compose]) -> None:
        # Do NOT use super.init(), though inherit all other aspects, else below attributes are typed in unwanted ways
        self.targets = targets
        self.data = data
        self.transforms = transforms
        self.target_transforms = target_transforms
        
    def __len__(self):
        return len(self.targets) 

    def __getitem__(self, index):
        """ 
        Returns tuple of image, single spurious feature for experiment, target
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = convert_img_axes(img, channel_first=False)
        img = np.asarray(img)
        img = Image.fromarray(img)

        if self.transforms is not None:
            img = self.transforms(img)

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return img, target

if __name__ == "__main__":
    my_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    og_ds = CIFAR10("./data", train=True,transform=my_transforms, target_transform=None, download=True)
    data = torch.as_tensor(og_ds.data) # This .data is natively in 0 to 1 range
    targets = torch.as_tensor(og_ds.targets).reshape(-1,1)
    my_ds = MyDataset(data=data, targets=targets, transforms=my_transforms, target_transforms= None)
    print(len(og_ds))
    print("Image\n",my_ds[0][0])
    print("Target\n",my_ds[0][1])

