import torch
from torch.utils.data import Dataset

from PIL import Image

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

        if self.transforms is not None:
            img = self.transforms(img)

        if self.target_transforms is not None:
            target = self.target_transforms(target)

        return self.data[index], self.targets[index].unsqueeze(0)

if __name__ == "__main__":
    pass