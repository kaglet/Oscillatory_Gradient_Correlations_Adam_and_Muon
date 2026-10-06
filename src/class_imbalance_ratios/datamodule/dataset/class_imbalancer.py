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

class TrainClassImbalancer():
    """
    Repeated pipeline with no assumptions on the input data, except targets used for indexing due to role created for task: class based sampling. 
    """
    def __init__(self, data: torch.Tensor, targets: torch.Tensor, extra_settings: dict) -> None:
        self._data, self._targets = data, targets
        self._experiment_ratio = extra_settings["experiment_ratio"] 
        
        len_data = len(self._data)
        new_idxs = np.random.permutation(len_data)

        self._targets = self._targets[new_idxs]

        self._ratios = self._set_sample_counts(self._targets)
        sampler = DeterministicUnderSampler(
            weighted_ratios=self._ratios, 
            targets=np.asarray(self._targets).astype(np.longlong)
        )

        self._chosen_mask, self._unchosen_mask = sampler.fit_resample()
        self._chosen_data = self.get_chosen_subset()

    @property 
    def class_ratios(self) -> dict[str, int]:
        ratios = {str(k):v for k,v in self._ratios.items()}
        return ratios
    
    def get_chosen_subset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        idxs = list(np.flatnonzero(self._chosen_mask).astype(int))
        return self._data[idxs], self._targets[idxs]

    def get_unchosen_subset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        idxs = list(np.flatnonzero(self._unchosen_mask).astype(int))
        return self._data[idxs], self._targets[idxs]
    
    def _set_sample_counts(self, targets) -> Dict[int, int]:
        """ 
        Outputs normalised weights dictionary from single down sampling ratio xor weight dict.
        If not assignable outputs with appropriate assertion errors violating assumptions. 
        """
        targets = np.asarray(targets, dtype=np.int16)
        labels = np.unique(targets)
        np.random.shuffle(labels)
        map_to_num = np.vectorize(lambda lbl: (targets == lbl).sum())
        totals = map_to_num(labels) 

        top_n = math.ceil(len(labels) / 2) # Half get applied the imbalance by default
        chosen_idxs = np.random.choice(labels, size=top_n, replace=False)
        final_factors = np.ones_like(labels, dtype=np.float32)
        final_factors[chosen_idxs] = np.divide(1, self._experiment_ratio)

        final_sample_counts = np.floor(np.multiply(totals, final_factors))

        labels.sort(axis=0)
        mapped = {int(key):int(val) for (key,val) in zip(labels,final_sample_counts)}

        return mapped 

class DatasetRenormalizer():
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

    def _get_data_stats(self, data)-> tuple[np.ndarray, np.ndarray]:
        """
        Assumes image data.
        """
        pix_sum = np.zeros(shape=(3,), dtype=np.float32)
        pix_sum_sq = np.zeros_like(pix_sum)
        # Access the *transformed* image via getter method
        img_dim = data[0][0].shape
        img_pix_count = img_dim[1] * img_dim[2]
        num_img = len(data)
        total_pix_count = num_img * img_pix_count
        for input, _ in data:
            input_np = np.asarray(input)
            pix_sum = np.sum([pix_sum, np.sum(input_np, axis=(1,2))], axis=0 )
            pix_sum_sq = np.sum([pix_sum_sq, np.sum(np.square(input_np), axis=(1,2))], axis=0) 

        global_mean = pix_sum / total_pix_count
        global_var = (pix_sum_sq / total_pix_count) - (np.square(global_mean))
        global_std = np.sqrt(global_var)

        return global_mean, global_std

    def renormed_transforms(self, apply_dset_norm: bool) -> transforms.Compose:
        new_transforms = self._og_transforms
        if apply_dset_norm:
            normalize = transforms.Normalize(
                mean=self._data_stats[0], # ImageNet standard norm
                std=self._data_stats[1]
            )
            new_transforms = transforms.Compose((self._og_transforms, normalize))

        return new_transforms

if __name__ == "__main__":
    my_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    og_ds = CIFAR10("/data", train=True,transform=my_transforms, target_transform=None, download=True)
    print(len(og_ds))
    data = torch.as_tensor(og_ds.data)
    targets = torch.as_tensor(og_ds.targets)
    settings = dict(experiment_ratio=20)
    imbalancer = TrainClassImbalancer(data=data, targets=targets, extra_settings=settings)
    chosen_subset = imbalancer.get_chosen_subset()
    unchosen_subset = imbalancer.get_unchosen_subset()
    print(len(chosen_subset[0]))
    print(len(unchosen_subset[0]))

    print("Target classes in unchosen subset", np.unique(np.asarray(unchosen_subset[1])))