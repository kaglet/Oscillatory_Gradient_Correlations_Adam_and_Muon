import logging
import math
import os
from pathlib import Path
import random
import numpy as np
from typing import Any, Callable, Dict, Literal, Optional, Type
from collections import Counter, namedtuple
from collections.abc import Mapping
import torch
import lightning as L
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torch.utils.data import Subset, random_split, DataLoader, TensorDataset
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
from src.base.datamodule.components.my_dataset import MyDataset
from src.class_imbalance_ratios.datamodule.components.deterministic_u_sampler import DeterministicUnderSampler

from PIL import Image

from src.class_imbalance_ratios.datamodule.dataset.class_imbalance_utils import ClassImbalancer, DataRenormalizer

__all__ = ["ClsImbDataModule"]

class ClsImbDataModule(CIFARDataModule):
    # TODO" Unify the data directory from env see old template
    def __init__(
        self,
        metadata_dir: str,
        train_transforms: transforms.Compose, val_transforms: transforms.Compose, test_transforms: transforms.Compose, 
        train_counts_map: Dict[int, int],
        val_counts_map: Dict[int, int],
        test_counts_map: Dict[int, int],
        dataset: Type[CIFAR10] | Type[CIFAR100] = CIFAR10, 
        data_dir: str = "/data", batch_size: int = 64, num_workers: int = 4,
    ):
        """
        * For the experiment set a ratio of R, where R % num_cifar_classes = 0
        * Half of the classes of CIFAR are set to the upperbound. The rest are divided by the ratio.
        * - weighted_ratios: Dict[int, float] accepts dynamic weights of length equivalent to CIFAR10 classes or CIFAR100 classes depending on the choice of kw arg. Weights will be normalised by proportion of total.
        * - downsampler_ratio: int, accepts a positive number, to apply strict logic of random 50% of classes chosen as dataset for 1 run of training (usage of this datamodule instance)
        * Exclusively one of the above two must be set, if both are set, dict input is discarded by convention. 
        """
        super().__init__(data_dir=data_dir, dataset=dataset ,train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, batch_size=batch_size, num_workers=num_workers)
        self._train_counts_map = train_counts_map
        self._val_counts_map = val_counts_map
        self._test_counts_map = test_counts_map

        self._metadata_dir = metadata_dir

        self._class_labels: list[str] = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck"
        ]

    def _get_masked_splits(self, data: torch.Tensor, targets: torch.Tensor, counts_map: dict[int, int]) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
            imbalancer = ClassImbalancer(class_distr=targets, counts_map=counts_map)
            chosen_mask = imbalancer.chosen_mask
            
            unchosen_mask = imbalancer.unchosen_mask

            return (data[chosen_mask], targets[chosen_mask]), (data[unchosen_mask], targets[unchosen_mask])

    def save_unused_metadata(self, counts_map: dict[int, int]):
            """
            Details data leftover after class imbalancer is used to select custom counts. 
            This can be used to denote data left over for own `test` OOD. 
            This need not have any relation to `test` data. 
            However some amount of data may be left from val, train, naturally per class that 
            could be used intelligently for a test distribution, to carry on the same pipeline. 
            Due to stochasticity in shuffle the same indices in the unused_mask, needs to be used per model/seed combo. 
            """
            torch.save(counts_map, self._metadata_dir + f"unused_class_counts.pt")

    def _save_count_metadata(self, stage: str, desired_counts_map: dict[str, int], actual_counts_map: dict[int, int]):
            """
            Desired map is from formatted labels. Actual shows what is stored in reality. 
            """
            print(f"Desired counts map:\n {desired_counts_map}")
            print(f"Actual counts map:\n {actual_counts_map}")
            os.makedirs(self._metadata_dir, exist_ok=True)
            torch.save(desired_counts_map, self._metadata_dir + f"{stage}_desired_class_counts.pt")
            torch.save(actual_counts_map, self._metadata_dir + f"{stage}_actual_class_counts.pt")

    def setup(self, stage: str)  -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            dset = self.ds("/data", train=True,transform=self.train_transforms, target_transform=None, download=True)
            targets = torch.as_tensor(dset.targets) # targets from CIFAR10 is list[Unknown] unfortunately, force into Tensor
            data = torch.as_tensor(dset.data)  
            
            # Randomly shuffle once and once only
            idxs = torch.randperm(data.shape[0])
            data = data[idxs]
            targets = targets[idxs]

            # Create train dataset
            (train_data, train_targets), (leftover_data, leftover_targets) = self._get_masked_splits(
                 data, 
                 targets, 
                 self._train_counts_map
            )
            self._train = MyDataset(train_data, train_targets, self.train_transforms, None)

            # Create val dataset 
            (val_data, val_targets), (unused_data, unused_targets) = self._get_masked_splits(
                 leftover_data, 
                 leftover_targets, 
                 self._val_counts_map
            )
            self._val = MyDataset(val_data, val_targets, self.val_transforms, None)

            # Apply custom normalization from train distribution
            train_renormer = DataRenormalizer(self.train_transforms, np.asanyarray(self._train.data))
            # Train receives appended version of og transforms stored by normalizer
            self._train.transforms = train_renormer.renormed_transforms(True)
            # Train receives only norm to append to its own transform pipeline
            self._val.transforms = transforms.Compose((self.val_transforms, train_renormer.norm_only))

            # Store metadata for train counts map
            train_desired_labels = {k:v for k,v in zip(self.class_labels, self._train_counts_map.values())}
            labels = train_targets.unique(return_counts=True, sorted=True)[0]
            counts = train_targets.unique(return_counts=True, sorted=True)[1]
            train_actual_labels = {int(l):int(c) for l,c in zip(labels, counts)}
            self._save_count_metadata("train", train_desired_labels, train_actual_labels)

            # Store metadata for val counts map
            val_desired_labels = {k:v for k,v in zip(self.class_labels, self._val_counts_map.values())}
            labels = val_targets.unique(return_counts=True, sorted=True)[0]
            counts = val_targets.unique(return_counts=True, sorted=True)[1]
            val_actual_labels = {int(l):int(c) for l,c in zip(labels, counts)}
            self._save_count_metadata("val", val_desired_labels, val_actual_labels)

            # Store metadata for unused counts map. Note there CANNOT be a "desired" map. See @`self.save_unused_metadata()`
            labels = unused_targets.unique(return_counts=True, sorted=True)[0]
            counts = unused_targets.unique(return_counts=True, sorted=True)[1]
            unused_actual_labels = {int(l):int(c) for l,c in zip(labels, counts)}
            self.save_unused_metadata(unused_actual_labels)

        # Ensure test is never mixed up with train, val for any model, and is seen once and once only at the end
        if stage == "test":
            self._test = self.ds(self.data_dir, train=False, transform=self.train_transforms, target_transform=None)

        if stage == "predict":
            self._predict = self.ds(self.data_dir, train=False, transform=self.train_transforms, target_transform=None)

    def class_ratios(self, stage: Literal["train"] | Literal["val"] | Literal["test"] | None) -> dict[int, int]: # remove getter to allow choice of stage
        if stage == "train":
             return self._train_counts_map
        elif stage == "val":
             return self._val_counts_map
        elif stage == "test":
             return self._test_counts_map

        # Below is redundant
        UserWarning(f"{stage} is detected as None, returning train ratios by default")
        return self._train_counts_map

    @property
    def class_labels(self) -> list[str]:
        return self._class_labels
    
# Entry point
if __name__ == "__main__":
    """
    Confirms smooth run of CIFAR10 dataloader, native image sizes, with minimal transforms
    """
    base_transforms = transforms.Compose([
        ToTensor(),
    ])

    class_labels = list(range(10))

    train_class_counts = [3500 for _ in range(10)]
    train_map = {k:v for k,v in zip(class_labels, train_class_counts)}

    val_class_counts = [1000 for _ in range(10)]
    val_map = {k:v for k,v in zip(class_labels, val_class_counts)}

    test_class_counts = [500 for _ in range(10)]
    test_map = {k:v for k,v in zip(class_labels, test_class_counts)}

    dmodule = ClsImbDataModule(
        num_workers=8, 
        val_counts_map=val_map,         
        train_counts_map=train_map, 
        test_counts_map=test_map,        
        train_transforms=base_transforms, 
        val_transforms=base_transforms, 
        test_transforms=base_transforms,
        metadata_dir=str(Path(__file__).parent)
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