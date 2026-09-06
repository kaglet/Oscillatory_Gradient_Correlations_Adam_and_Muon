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
from tqdm import tqdm

from src.base.datamodule.cifar_datamodule import CIFARDataModule
from src.class_imbalance_ratios.datamodule.components.deterministic_u_sampler import DeterministicUnderSampler

__all__ = ["ClsImbDataModule"]

class ClsImbDataModule(CIFARDataModule):
    # TODO" Unify the data directory from env see old template
    def __init__(
        self,
        weighted_ratios: Dict[int, float] | None = None,
        downsampler_ratio: int = 20,
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
        self.experiment_ratio = downsampler_ratio 

    def setup(self, stage: str)  -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self._cifar_full = self.ds(self.data_dir, train=True, transform=self.train_transforms)
            self._original_shape = (len(self._cifar_full), *(self._cifar_full[0][0].shape))
            sampler = DeterministicUnderSampler(weighted_ratios=self._set_sample_counts(self._cifar_full), data= np.asarray(self._cifar_full.data.reshape(self._original_shape)), targets=np.asarray(self._cifar_full.targets))
            
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

    def _set_sample_counts(self, ds) -> Dict[int, int]:
        """ 
        Outputs normalised weights dictionary from single down sampling ratio xor weight dict.
        If not assignable outputs with appropriate assertion errors violating assumptions. 
        """
        targets = np.asarray(ds.targets, dtype=np.int16)
        labels = np.unique(targets)
        labels_data = np.unique(targets, return_counts=True)
        np.random.shuffle(labels)
        map_to_num = np.vectorize(lambda lbl: (targets == lbl).sum())
        totals = map_to_num(labels) 

        top_n = math.ceil(len(labels) / 2)
        chosen_idxs = np.random.choice(labels, size=top_n, replace=False)
        final_factors = np.ones_like(labels, dtype=np.float32)
        final_factors[chosen_idxs] = np.divide(1, self.experiment_ratio)

        final_sample_counts = np.floor(np.multiply(totals, final_factors))

        labels.sort(axis=0)
        mapped = {int(key):int(val) for (key,val) in zip(labels,final_sample_counts)}
        labels_data = np.unique(targets, return_counts=True)

        return mapped 

    def get_train_data_stats(self):
        # Stick to single type, numpy OR torch 
        # Test numpy
        train_ds = self._cifar_full.data[self._cifar_train.indices]
        train_shape = len(train_ds), self._original_shape[1], self._original_shape[2], self._original_shape[3]
        train_ds = train_ds.reshape(train_shape)
        pix_sum = np.zeros(shape=(3,), dtype=np.float32)
        pix_sum_sq = np.zeros_like(pix_sum)
        img_dim = self._cifar_train[0][0].shape
        img_pix_count = img_dim[1] * img_dim[2]
        num_img = len(self._cifar_train)
        total_pix_count = num_img * img_pix_count
        for input in train_ds:
            np_input = np.array(input, dtype=np.float32)
            pix_sum = np.sum([pix_sum, np.sum(np_input, axis=(1,2))], axis=0 )
            pix_sum_sq = np.sum([pix_sum_sq, np.sum(np.square(np_input), axis=(1,2))], axis=0) 

        total_mean = pix_sum / total_pix_count
        total_var = (pix_sum_sq / total_pix_count) - (np.square(total_mean))
        total_std = np.sqrt(total_var)

        return total_mean, total_std

# Entry point
if __name__ == "__main__":
    """
    Confirms smooth run of dataloader, native image sizes, with minimal transforms
    """
    dmodule = ClsImbDataModule(downsampler_ratio=20) 
    dmodule.prepare_data()
    dmodule.setup("fit")
    print(f"Train dataloader length: {len(dmodule.train_dataloader())}")
    print(f"Stats mean, std per RGB channel {dmodule.get_train_data_stats()}")
    # for input, target in dmodule.train_dataloader():
    #     print(f"Inputs of len {len(input)} targets of len {len(target)}")
    #     print(target)