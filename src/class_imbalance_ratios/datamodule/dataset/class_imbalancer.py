# Controls the ratios innately simply with a reindexing function, with the same core dataset
# A randomizer helps because the dataset core is not touched as on demand core except as a map, and randomness can use the seed
import math
from pathlib import Path
from typing import Any, Callable, Dict
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import random_split, TensorDataset

from src.class_imbalance_ratios.datamodule.components.deterministic_u_sampler import DeterministicUnderSampler
class ClassImbalancerSmall(CIFAR10):
    def __init__(self, extra_settings: dict, root: str | Path, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)

class ClassImbalancerLarge(CIFAR100):
    def __init__(self, extra_settings: dict, root: str | Path, train: bool = True, transform: Callable[..., Any] | None = None, target_transform: Callable[..., Any] | None = None, download: bool = False) -> None:
        super().__init__(root, train, transform, target_transform, download)
        self.experiment_ratio = extra_settings["experiment_ratio"] 
        # TODO: Do split right here, before resampling, have full data to extract and construct sufficient val set already
        percent_train = 0.8
        train_range = np.arange(0, math.ceil(len(self.data)*percent_train))
        val_range = np.arange(math.ceil(len(self.data)*percent_train), len(self.data))

        train_idx = np.random.permutation(train_range)
        self._val_idx = np.random.permutation(val_range)

        self.targets = self.targets[train_idx]
        sampler = DeterministicUnderSampler(
            weighted_ratios=self._set_sample_counts(np.asarray(self.targets, dtype=np.int16)), 
            data = np.asarray(self.data), # .data gets raw data not <=> with getitem accessor purpose for PIL image usually 
            targets=np.asarray(self.targets)
        )
        self.data, self.targets = sampler.fit_resample()

        # TODO: Append these stats to EXISTING transforms similar to how I had done it but reference parent class, this way it is innate 
        self._mean, self._std = self.get_train_data_stats()

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
    def val_data(self) -> np.ndarray:
        """ 
        Return validation data as TODO Tensor Datase, different from the main train .data as default.
        """
        return self.data[self._val_idx]
    
    @property 
    def mean(self) -> np.ndarray:
        return self._mean 

    @property 
    def std(self) -> np.ndarray:
        return self._std 

    def get_train_data_stats(self)-> tuple[np.ndarray, np.ndarray]:
        pix_sum = np.zeros(shape=(3,), dtype=np.float32)
        pix_sum_sq = np.zeros_like(pix_sum)
        # Access the transformed image via getter method
        img_dim = self.data[0][0].shape
        img_pix_count = img_dim[1] * img_dim[2]
        num_img = len(self.data)
        total_pix_count = num_img * img_pix_count
        for input in self.data:
            input_np = np.asarray(input)
            pix_sum = np.sum([pix_sum, np.sum(input_np, axis=(1,2))], axis=0 )
            pix_sum_sq = np.sum([pix_sum_sq, np.sum(np.square(input_np), axis=(1,2))], axis=0) 

        global_mean = pix_sum / total_pix_count
        global_var = (pix_sum_sq / total_pix_count) - (np.square(global_mean))
        global_std = np.sqrt(global_var)

        return global_mean, global_std