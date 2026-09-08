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
            self._cifar_full = self.ds(self.data_dir, train=True)
            sampler = DeterministicUnderSampler(
                weighted_ratios=self._set_sample_counts(self._cifar_full), 
                data = np.asarray(self._cifar_full.data), 
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
            train_stats = self.get_ds_stats(train_data)
            normalize = transforms.Normalize(
                mean=train_stats[0], # ImageNet standard norm
                std=train_stats[1]
            )
            self.train_transforms.transforms.append(normalize)
            self._cifar_train = CIFAR10Helper(train_data, train_target, transform=self.train_transforms, target_transform=None)

        # Ensure test is never mixed up with train, val for any model, and is seen once and once only at the end
        if stage == "test":
            self._cifar_test = self.ds(self.data_dir, train=False)

        if stage == "predict":
            self._cifar_predict = self.ds(self.data_dir, train=False)

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
    def get_ds_stats(self, train_ds):
        # We stick to a single type: Numpy. To switch to pytorch style vectors or array switch ALL code for ease of interpretability and discourage mixing. 
        pix_sum = np.zeros(shape=(3,), dtype=np.float32)
        pix_sum_sq = np.zeros_like(pix_sum)
        img_dim = train_ds[0][0].shape
        img_pix_count = img_dim[0] * img_dim[1]
        num_img = len(train_ds)
        total_pix_count = num_img * img_pix_count
        for input in train_ds:
            np_input = np.array(input, dtype=np.float32)
            pix_sum = np.sum([pix_sum, np.sum(np_input, axis=(0,1))], axis=0 )
            pix_sum_sq = np.sum([pix_sum_sq, np.sum(np.square(np_input), axis=(0,1))], axis=0) 

        total_mean = pix_sum / total_pix_count
        total_var = (pix_sum_sq / total_pix_count) - (np.square(total_mean))
        total_std = np.sqrt(total_var)

        return total_mean, total_std
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._cifar_train, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._cifar_val, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._cifar_test, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=True)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self._cifar_predict, batch_size=self.batch_size_per_device, num_workers=self.num_workers, persistent_workers=True)
    
# Entry point
if __name__ == "__main__":
    """
    Confirms smooth run of dataloader, native image sizes, with minimal transforms
    """
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], # ImageNet standard norm
    #     std=[0.229, 0.224, 0.225]
    # )
    base_transforms = transforms.Compose([
        ToTensor(),
    ])

    settings = dict(
        train_transforms=base_transforms, 
        val_transforms=base_transforms, 
        test_transforms=base_transforms
    )
    dmodule = ClsImbDataModule(downsampler_ratio=20, num_workers=1, weighted_ratios=None, **settings) 
    dmodule.prepare_data()
    dmodule.setup("fit")
    print(f"Train dataloader length: {len(dmodule.train_dataloader())}")

    i = 0
    for inputs, targets in dmodule.train_dataloader():
        # print(f"Inputs of len {len(inputs)} targets of len {len(targets)}")
        # print(inputs, targets)
        img = convert_img_axes(inputs[0])
        # img = to_pil_image(img, mode="RGB")
        # img = base_transforms(img)
        viz_img(img, label=targets[0])
        i += 1
        if i == 10: 
            break

    