import logging
import math
import os
import random
import numpy as np
from typing import Any, Callable, Dict, Literal, Optional, Type
from collections import Counter
from collections.abc import Mapping
import torch
import lightning as L
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import to_pil_image
from imblearn.under_sampling import RandomUnderSampler
from tqdm import tqdm

from src.base.datamodule.cifar_datamodule import CIFAR10DataModule, CIFARDataModule
from src.base.datamodule.components.cifar_10_helper import CIFAR10Helper
from src.base.datamodule.components.convert_ax import convert_img_axes
from src.base.datamodule.components.img_viz import viz_img
from src.class_imbalance_ratios.datamodule.components.deterministic_u_sampler import DeterministicUnderSampler

from PIL import Image

from src.class_imbalance_ratios.datamodule.dataset.class_imbalancer import TrainClassImbalancer

__all__ = ["ClsImbDataModule"]

class ClsImbDataModule(CIFARDataModule):
    # TODO" Unify the data directory from env see old template
    def __init__(
        self,
        train_transforms: transforms.Compose, val_transforms: transforms.Compose, test_transforms: transforms.Compose, dataset: Type[CIFAR10] | Type[CIFAR100] = CIFAR10, data_dir: str = "/data", batch_size: int = 64, num_workers: int = 4,
        weighted_ratios: Dict[int, float] | None = None,
        downsampler_ratio: int = 20,
    ):
        """
        * For the experiment set a ratio of R, where R % num_cifar_classes = 0
        * Half of the classes of CIFAR are set to the upperbound. The rest are divided by the ratio.
        * - weighted_ratios: Dict[int, float] accepts dynamic weights of length equivalent to CIFAR10 classes or CIFAR100 classes depending on the choice of kw arg. Weights will be normalised by proportion of total.
        * - downsampler_ratio: int, accepts a positive number, to apply strict logic of random 50% of classes chosen as dataset for 1 run of training (usage of this datamodule instance)
        * Exclusively one of the above two must be set, if both are set, dict input is discarded by convention. 
        """
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, batch_size=batch_size, num_workers=num_workers)
        if downsampler_ratio is not None and weighted_ratios is not None: 
            weighted_ratios = None 

        self._target_weights = weighted_ratios
        self.experiment_ratio = downsampler_ratio 

    def setup(self, stage: str)  -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            dset = self.ds("/data", train=True,transform=self.train_transforms, target_transform=None, download=True)
            settings = dict(experiment_ratio=20)
            imbalancer = TrainClassImbalancer(dataset=dset, extra_settings=settings)
            chosen_subset = imbalancer.get_chosen_subset(apply_dset_norm=True)
            unchosen_subset = imbalancer.get_unchosen_subset()
            self._train = chosen_subset
            self._val = unchosen_subset
            
        # Ensure test is never mixed up with train, val for any model, and is seen once and once only at the end
        if stage == "test":
            self._test = self.ds(self.data_dir, train=False, transform=self.train_transforms, target_transform=None)

        if stage == "predict":
            self._predict = self.ds(self.data_dir, train=False, transform=self.train_transforms, target_transform=None)
    
# Entry point
if __name__ == "__main__":
    """
    Confirms smooth run of CIFAR10 dataloader, native image sizes, with minimal transforms
    """
    base_transforms = transforms.Compose([
        ToTensor(),
    ])

    dmodule = ClsImbDataModule(
        downsampler_ratio=20, 
        num_workers=8, 
        weighted_ratios=None,         
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
        viz_img(img, label=targets[0])
        i += 1
        if i == 10: 
            break