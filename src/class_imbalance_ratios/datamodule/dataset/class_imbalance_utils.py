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
from src.class_imbalance_ratios.datamodule.components.custom_subset import TupledSubset
from src.class_imbalance_ratios.datamodule.components.deterministic_u_sampler import DeterministicUnderSampler
import torch.nn as nn

class ReshuffleCountMap():
    """
    From original map get a remap with shuffled values. 
    Consistent  input -> output types prevents confusing type conversions. 
    This is stochastic shuffling dictated by the torch manual seed. 

    """
    def __init__(self, counts_map: dict[int, int]) -> None:
        self._counts_map = counts_map
        counts = torch.as_tensor(list(counts_map.values()))
        idxs = torch.randperm(counts.shape[0]) # torch lacks simple np.shuffle like function. 
        self._new_counts = counts[idxs]

    @property
    def remapped(self) -> dict[int, int]:
        new_counts_map = {k:int(v) for k,v in zip(self._counts_map.keys(), self._new_counts)}
        return new_counts_map

class ClassImbalancer():
    """
    For agnostic rebalancing take only class distribution, absolute class count mapping. 
    Return a mask of indices of distribution chosen and unchosen "leftovers".
    Same order of presentation is returned. No shuffling is performed. 
    This is so the mask does not become uncertain relative to the input data given, and can be reused. 
    This helps tracking, as shuffling should only be done by a dedicated class. 
    """
    def __init__(self, class_distr: torch.Tensor, counts_map: dict[int, int]) -> None:
        # Check class dist has all classes in map to set new counts of, else abort
        sorted_mapped_classes = torch.as_tensor(list(counts_map.keys())).unique(sorted=True) 
        sorted_distribution_classes = class_distr.unique(sorted=True)
        assert (torch.equal(sorted_distribution_classes, sorted_mapped_classes)), f"Mismatch detected. Class map of {sorted_mapped_classes} cannot be applied to distribution classes {sorted_distribution_classes}."

        self._class_distribution = class_distr
        self._counts_map = counts_map
        self._chosen_mask, self._unchosen_mask = self._get_masks()

    def _get_masks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        targets = np.asarray(self._class_distribution).reshape((-1,1))
  
        chosen_mask = np.full_like(targets, False, dtype=np.bool)

        for (lbl, count) in self._counts_map.items():
            lbl_matcher = (targets == lbl)
            assert type(lbl_matcher) is np.ndarray, "lbl_matcher should be type np.ndarray"
            # Negatives do not contribute to cumsum and are detected by first lbl_matcher using AND - &
            is_in_scope = (lbl_matcher.cumsum()<=count).reshape(-1,1)
            lbl_matcher = lbl_matcher & is_in_scope

            chosen_mask = chosen_mask | lbl_matcher

        """
        # For sanity/debug checking chosen mask [0] should always be True 
        # while [-1] is false, and vice versa for unchosen mask. 
        # This is due to the cumsum and no shuffling.
        """
        unchosen_mask = ~chosen_mask 
        return torch.as_tensor(chosen_mask).squeeze(), torch.as_tensor(unchosen_mask).squeeze() # Masks must match shape at dim[0] for indexed tensors

    @property
    def chosen_mask(self) -> torch.Tensor:
        return self._chosen_mask

    @property
    def chosen_idxs(self) -> torch.Tensor:
        idxs = torch.nonzero(self._chosen_mask)
        return idxs

    @property
    def unchosen_mask(self) -> torch.Tensor:
        return self._unchosen_mask

    @property
    def unchosen_idxs(self) -> torch.Tensor:
        idxs = torch.nonzero(self._unchosen_mask)
        return idxs

class DataRenormalizer():
    """
    In pipelines creating subsets of data, calculates the mean, std based of argument passed in data. 
    Based of dataset stats can obtain mean, std for internal use.
    Main use is to append `torch.transforms.Normalize` to original transforms as `og_transforms` of data.
    Assumes existing transforms DO NOT include a Normalize based off the statistics of the dataset. 
    In the case this is not adhered to, dataset will be normalized once, instead of from the final attributes of the newly processed dataset. 
    """
    def __init__(self, og_transforms: transforms.Compose, data: np.ndarray) -> None:
        self._og_transforms = og_transforms
        self._data = data
        self._data_stats = self._get_data_stats(self._data)

    def _get_data_stats(self, data: np.ndarray)-> tuple[np.ndarray, np.ndarray]:
        """
        Assumes image data.
        """
        pix_sum = np.zeros(shape=(3,), dtype=np.float32)
        pix_sum_sq = np.zeros_like(pix_sum)
        # Access the *transformed* image via getter method
        img_dim = data[0].shape
        img_pix_count: int = img_dim[0] * img_dim[1]
        num_img = len(data)
        total_pix_count = num_img * img_pix_count
        for input in data:
            input_np = np.asarray(input)
            pix_sum = np.sum([pix_sum, np.sum(input_np, axis=(0,1))], axis=0 )
            pix_sum_sq = np.sum([pix_sum_sq, np.sum(np.square(input_np), axis=(0,1))], axis=0) 

        global_mean = pix_sum / total_pix_count
        global_var = (pix_sum_sq / total_pix_count) - (np.square(global_mean))
        global_std = np.sqrt(global_var)

        return global_mean, global_std

    def renormed_transforms(self, output_new: bool) -> transforms.Compose:
        new_transforms = self._og_transforms
        if output_new:
            normalize = transforms.Normalize(
                mean=self._data_stats[0], # ImageNet standard norm
                std=self._data_stats[1]
            )
            new_transforms = transforms.Compose((self._og_transforms, normalize))

        return new_transforms

    @property
    def norm_only(self) -> transforms.Normalize:
        normalize = transforms.Normalize(
            mean=self._data_stats[0], # ImageNet standard norm
            std=self._data_stats[1]
        )

        return normalize

if __name__ == "__main__":
    my_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    og_ds = CIFAR10("/data", train=True,transform=my_transforms, target_transform=None, download=True)
    print(len(og_ds))
    data = torch.as_tensor(og_ds.data)
    targets = torch.as_tensor(og_ds.targets).reshape(-1,1)
    class_counts = [200 for _ in range(10)]
    class_labels = list(range(10))
    class_counts_map = {k:v for k,v in zip(class_labels, class_counts)}
    imbalancer = ClassImbalancer(class_distr=targets, counts_map=class_counts_map)
    chosen_mask = imbalancer.chosen_mask
    unchosen_mask = imbalancer.unchosen_mask

    count_chosen = chosen_mask.sum()
    count_unchosen = unchosen_mask.sum()
    print(count_chosen)
    print(count_unchosen)

    print("Target classes in unchosen subset (label and count)\n", targets[unchosen_mask].unique(return_counts=True))
    print("Target classes in chosen subset (label and count)\n", targets[chosen_mask].unique(return_counts=True))