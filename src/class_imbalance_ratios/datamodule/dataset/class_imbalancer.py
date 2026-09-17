# Controls the ratios innately simply with a reindexing function, with the same core dataset
# A randomizer helps because the dataset core is not touched as on demand core except as a map, and randomness can use the seed
import math
from pathlib import Path
from typing import Any, Callable, Dict, Type
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import random_split, TensorDataset
import torchvision.transforms as transforms

from src.class_imbalance_ratios.datamodule.components.deterministic_u_sampler import DeterministicUnderSampler

class TrainClassImbalancer():
    def __init__(self, dataset: Type[CIFAR10] | Type[CIFAR100], extra_settings: dict, root: str | Path, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        self.ds = dataset(root, train, transform, target_transform, download)
        self.data, self.targets = self.ds.data, self.ds.targets
        self.experiment_ratio = extra_settings["experiment_ratio"] 
        # TODO: Do split right here, before resampling, have full data to extract and construct sufficient val set already
        data_range = len(self.data)
        new_idxs = np.random.permutation(data_range)

        self.targets = np.asarray(self.targets[new_idxs], dtype=np.longlong)

        sampler = DeterministicUnderSampler(
            weighted_ratios=self._set_sample_counts(self.targets), 
            targets=self.targets
        )
        subset_idxs, unchosen_idxs = sampler.fit_resample()
        self.subset_data, self.subset_targets = (self.data[subset_idxs], 
                                self.targets[subset_idxs])

        # TODO: Append these stats to EXISTING transforms similar to how I had done it but reference parent class, this way it is innate 
        self._mean, self._std = self.get_train_data_stats(self.subset_data)
        normalize = transforms.Normalize(
            mean=self._mean, # ImageNet standard norm
            std=self._std
        )
        self.ds.transforms = transforms.Compose([self.ds.transforms, normalize])
        # When used with subset idxs defines 

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
        final_factors[chosen_idxs] = np.divide(1, self.experiment_ratio)

        final_sample_counts = np.floor(np.multiply(totals, final_factors))

        labels.sort(axis=0)
        mapped = {int(key):int(val) for (key,val) in zip(labels,final_sample_counts)}

        return mapped 

    @property
    def train_subset(self) -> :
        return self.subset_data, self.subset_targets
    
    def get_train_data_stats(self, data)-> tuple[np.ndarray, np.ndarray]:
        pix_sum = np.zeros(shape=(3,), dtype=np.float32)
        pix_sum_sq = np.zeros_like(pix_sum)
        # Access the transformed image via getter method
        img_dim = data[0][0].shape
        img_pix_count = img_dim[0] * img_dim[1]
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

class TrainClassImbalancerSmall(TrainClassImbalancer):
    def __init__(self, dataset:  Type[CIFAR10] | Type[CIFAR100], extra_settings: Dict, root: str | Path, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        super().__init__(dataset, extra_settings, root, train, transform, target_transform, download)