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

__all__ = ["ClsImbDataModule"]

class ClsImbDataModule(CIFARDataModule):
    # TODO" Unify the data directory from env see old template
    def __init__(
        self,
        train_transforms: transforms.Compose, val_transforms: transforms.Compose, test_transforms: transforms.Compose, dataset: Type[CIFAR10] | Type[CIFAR100] = CIFAR10, data_dir: str = "/data", batch_size: int = 2048, num_workers: int = 8,
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
        super().__init__(train_transforms, val_transforms, test_transforms)
        if downsampler_ratio is not None and weighted_ratios is not None: 
            weighted_ratios = None 

        self._target_weights = weighted_ratios
        self.experiment_ratio = downsampler_ratio 

    def setup(self, stage: str)  -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self._cifar_full = self.ds(self.data_dir, train=True)
            sampler = DeterministicUnderSampler(
                weighted_ratios=self._set_sample_counts(self._cifar_full), 
                data = np.asarray(self._cifar_full.data), # .data gets raw data not <=> with getitem accessor purpose for PIL image usually 
                targets=np.asarray(self._cifar_full.targets)
            )
            # Returns subset of dataset of under sampler
            resampled_subset = sampler.fit_resample()
            train, val = random_split(
                resampled_subset, [0.8, 0.2]
            )
            # TODO: Do same process for train data
            chosen_idxs = train.indices
            train_data = np.asarray(resampled_subset.data[chosen_idxs])
            train_target = np.asarray(resampled_subset.targets[chosen_idxs])
            no_norm_train = CIFAR10Helper(train_data, train_target, transform=self.train_transforms, target_transform=None)
            # Learn how stats will look like if we apply all transforms except norm or any stat based transforms
            train_stats = self.get_ds_data_stats(no_norm_train)
            mean = np.ndarray.tolist(train_stats[0])
            std = np.ndarray.tolist(train_stats[1])

            normalize = transforms.Normalize(
                mean=mean, # ImageNet standard norm
                std=std
            )
            self.train_transforms.transforms.append(normalize)

            # Apply og transforms then stat based transforms learned on og transformed data
            norm_train = CIFAR10Helper(train_data, train_target, transform=self.train_transforms, target_transform=None)
            self._cifar_train = norm_train

            chosen_idxs = val.indices
            val_data = np.asarray(resampled_subset.data[chosen_idxs])
            val_target = np.asarray(resampled_subset.targets[chosen_idxs])

            # Apply og transforms then stat based transforms learned on og transformed data
            norm_train = CIFAR10Helper(val_data, val_target, transform=self.train_transforms, target_transform=None)
            self._cifar_val = norm_train

        # Ensure test is never mixed up with train, val for any model, and is seen once and once only at the end
        if stage == "test":
            self._cifar_test = self.ds(self.data_dir, train=False, transform=self.train_transforms, target_transform=None)

        if stage == "predict":
            self._cifar_predict = self.ds(self.data_dir, train=False, transform=self.train_transforms, target_transform=None)

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

    # TODO: Set .train_transforms post init during setup call the get_stats for each current train, val, test splits
    def get_ds_data_stats(self, train_data):
        # We stick to a single type: Numpy. To switch to pytorch style vectors or array switch ALL code for ease of interpretability and discourage mixing. 
        pix_sum = np.zeros(shape=(3,), dtype=np.float32)
        pix_sum_sq = np.zeros_like(pix_sum)
        # Access the transformed image via getter method
        img_dim = train_data[0][0].shape
        img_pix_count = img_dim[1] * img_dim[2]
        num_img = len(train_data)
        total_pix_count = num_img * img_pix_count
        for input, _ in train_data:
            input_np = np.asarray(input)
            pix_sum = np.sum([pix_sum, np.sum(input_np, axis=(1,2))], axis=0 )
            pix_sum_sq = np.sum([pix_sum_sq, np.sum(np.square(input_np), axis=(1,2))], axis=0) 

        total_mean = pix_sum / total_pix_count
        total_var = (pix_sum_sq / total_pix_count) - (np.square(total_mean))
        total_std = np.sqrt(total_var)

        return total_mean, total_std
    
# Entry point
if __name__ == "__main__":
    """
    Confirms smooth run of CIFAR10 dataloader, native image sizes, with minimal transforms
    """
    base_transforms = transforms.Compose([
        ToTensor(),
    ])

    settings = dict(

    )
    dmodule = ClsImbDataModule(downsampler_ratio=20, num_workers=1, weighted_ratios=None,         train_transforms=base_transforms, 
        val_transforms=base_transforms, 
        test_transforms=base_transforms) 
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

# Todo: Perform this as experiments via config, use of ratio and other settings, move ImageNet norm to other datasets not this one. 