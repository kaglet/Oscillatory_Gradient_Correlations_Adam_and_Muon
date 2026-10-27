# Controls the ratios innately simply with a reindexing function, with the same core dataset
# A randomizer helps because the dataset core is not touched as on demand core except as a map, and randomness can use the seed
import math
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Type
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Subset, random_split, TensorDataset
import torchvision.transforms as transforms
import torch
from src.base.datamodule.components.my_dataset import MyDataset
from src.class_imbalance_ratios.datamodule.components.custom_subset import TupledSubset
import torch.nn as nn
from PIL import Image

from src.spurious_pix.datamodule.components.pixel_injection_map import PixelInjectionMap

class PixelInjector():
    """

    """
    def __init__(self, dataset: MyDataset, injection_map: PixelInjectionMap, intensity: int) -> None:
        # Check class dist has all classes in map to set new counts of, else abort
        self._ds = dataset
        for lbl, loc in injection_map.injections_mapping.items():
            mask = self._ds.targets == int(lbl)
            class_len = len(self._ds.data[mask])
            for idx in range(class_len):
                PIL_sample = self._ds[idx][0]
                _ = self._inject_pix(PIL_sample, list(loc), intensity)

    def _inject_pix(self, sample: Any, loc: list, intensity: int) -> Any:
        x, y = loc[0], loc[1]
        sample[x,y] = intensity
        return sample[x,y]

    @property
    def injected_ds(self):
        return self._ds

if __name__ == "__main__":
    my_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    og_ds = CIFAR10("./data", train=True,transform=my_transforms, target_transform=None, download=True)
    