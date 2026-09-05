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
from torchvision.transforms import v2, ToTensor

from imblearn.under_sampling import RandomUnderSampler

from src.base.datamodule.cifar_datamodule import CIFARDataModule
from src.class_imbalance_ratios.datamodule.components.deterministic_u_sampler import DeterministicUnderSampler

__all__ = ["ClsImbDataModule"]

class ClsImbDataModule(CIFARDataModule):
    # TODO" Unify the data directory from env see old template
    def __init__(
        self,
        weighted_ratios: Dict[int, float] | None = None,
        downsampler_ratio: int | None = None,
        *args: Any, 
        **kwargs: Any
    ):
        """
        * For the experiment set a ratio of R, where R % num_cifar_classes = 0
        * Half of the classes of CIFAR are set to the upperbound. The rest are divided by the ratio.
        * - weighted_ratios: Dict[int, float] accepts dynamic weights of length equivalent to CIFAR10 classes or CIFAR100 classes depending on the choice of kw arg. Weights will be normalised by proportion of total.
        * - downsampler_ratio: int, accepts a positive number, to apply strict logic of random 50% of classes chosen as dataset for 1 run of training (usage of this datamodule instance)
        * Exclusively one of the above two must be set, if both are set, dict input is discarded by convention. 
        """
        super().__init__(*args, **kwargs)
        if downsampler_ratio is not None and weighted_ratios is not None: 
            weighted_ratios = None 

        self._target_weights = weighted_ratios
        self._downsampler_ratio = downsampler_ratio 

    def setup(self, stage: str)  -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar_full = self.ds(self.data_dir, train=True, transform=self.train_transforms)
            sampler = DeterministicUnderSampler(weighted_ratios=self._get_ratios(cifar_full), data= np.asarray(cifar_full.data), targets=np.asarray(cifar_full.targets))
            
            # Returns subset of dataset 
            resampled_subset = sampler.fit_resample()
            self._cifar_train, self._cifar_val = random_split(
                resampled_subset, [0.8, 0.2]
            )

        # Ensure test is never mixed up with train, val for any model, and is seen once and once only at the end
        if stage == "test":
            self._cifar_test = self.ds(self.data_dir, train=False, transform=self.test_transforms)

        if stage == "predict":
            self._cifar_predict = self.ds(self.data_dir, train=False, transform=self.test_transforms)

    def _get_ratios(self, ds, labels=range(10)) -> Dict[int, int]:
        """ 
        Outputs normalised weights dictionary from single down sampling ratio xor weight dict.
        If not assignable outputs with appropriate assertion errors violating assumptions. 
        """
        len_ds = len(ds) - 1
        len_labels = len(ds.targets) - 1
        final_ratios = None

        if self._target_weights is not None:
            total = sum([val for val in self._target_weights.values()])
            assert total == len_labels, f"Dict given has len {len_labels} != target domain len {len_ds}"
            final_ratios = {i:math.ceil((r/total)*len_ds) for (i,r) in self._target_weights.items()}
        elif self._downsampler_ratio is not None:
            class_ratios = [(i,1) for i in range(len_ds)]
            random.shuffle(class_ratios)
            slice_first = class_ratios[math.ceil(len_ds / 2) : ]
            slice_first = {i:1 for (i,r) in slice_first}

            slice_last = class_ratios[ : math.ceil(len_ds / 2)]
            slice_last = {i:self._downsampler_ratio for (i,r) in slice_last}
            final_ratios = {**slice_first, **slice_last}

        assert final_ratios is not None, "Final weights not assigned, aborting"

        return final_ratios

# Entry point
if __name__ == "__main__":
    """
    Confirms smooth run of dataloader, native image sizes, with minimal transforms
    """
    import warnings

    warnings.filterwarnings("ignore", "*pkg_resources is deprecated*")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet standard norm
                                    std=[0.229, 0.224, 0.225])
    # TODO: Resize images smaller and train on subset for efficiency and MVP.
    celeb_a_transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        ToTensor(),
        normalize
    ])
    
    dmodule = ClsImbDataModule(downsampler_ratio=20) 
    dmodule.prepare_data()
    dmodule.setup("fit")
    print(f"Train dataloader length: {len(dmodule.train_dataloader())}")

    for input, target in dmodule.train_dataloader():
        print(f"Inputs of len {len(input)} targets of len {len(target)}")
        print(target)