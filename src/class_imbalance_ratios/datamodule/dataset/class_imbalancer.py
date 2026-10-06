# Controls the ratios innately simply with a reindexing function, with the same core dataset
# A randomizer helps because the dataset core is not touched as on demand core except as a map, and randomness can use the seed
import math
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Type
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Subset, random_split, TensorDataset
import torchvision.transforms as transforms

from src.class_imbalance_ratios.datamodule.components.custom_subset import TupledSubset
from src.class_imbalance_ratios.datamodule.components.deterministic_u_sampler import DeterministicUnderSampler

class TrainClassImbalancer():
    def __init__(self, dataset: CIFAR10 | CIFAR100, extra_settings: dict) -> None:
        self._dset = dataset # TODO: Remove assumption of .data, .targets from data input which may be Subset
        self._data, self._targets = self._dset.data, np.asarray(self._dset.targets, dtype=np.longlong)
        self._experiment_ratio = extra_settings["experiment_ratio"] 
        # TODO: Do split right here, before resampling, have full data to extract and construct sufficient val set already
        len_data = len(self._data)
        new_idxs = np.random.permutation(len_data)

        self._targets = self._targets[new_idxs]

        self._ratios = self._set_sample_counts(self._targets)
        sampler = DeterministicUnderSampler(
            weighted_ratios=self._ratios, 
            targets=self._targets
        )

        self._chosen_mask, self._unchosen_mask = sampler.fit_resample()
        self._og_transforms = self._dset.transforms
        self._chosen_data = self.get_chosen_subset(apply_dset_norm=False)
        stats = self._get_data_stats(self._chosen_data)
        self._chosen_data_stats = stats

    @property 
    def class_ratios(self) -> dict[str, int]:
        ratios = {str(k):v for k,v in self._ratios.items()}

        return ratios
    
    def get_chosen_subset(self, apply_dset_norm:bool=False) -> TupledSubset:
        if apply_dset_norm:
            normalize = transforms.Normalize(
                mean=self._chosen_data_stats[0], # ImageNet standard norm
                std=self._chosen_data_stats[1]
            )
            self._dset.transforms = transforms.Compose((self._og_transforms, normalize))
        else:
            self._dset.transforms = self._og_transforms

        idxs = list(np.flatnonzero(self._chosen_mask).astype(int))

        subset = TupledSubset(self._dset, idxs, self._dset.targets)

        return subset

    def get_unchosen_subset(self) -> Subset:
        idxs = list(np.flatnonzero(self._unchosen_mask).astype(int))
        return Subset(self._dset, idxs)
    
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

    def _get_data_stats(self, data)-> tuple[np.ndarray, np.ndarray]:
        pix_sum = np.zeros(shape=(3,), dtype=np.float32)
        pix_sum_sq = np.zeros_like(pix_sum)
        # Access the *transformed* image via getter method
        img_dim = data[0][0].shape
        img_pix_count = img_dim[1] * img_dim[2]
        num_img = len(data)
        total_pix_count = num_img * img_pix_count
        for input, target in data:
            input_np = np.asarray(input)
            pix_sum = np.sum([pix_sum, np.sum(input_np, axis=(1,2))], axis=0 )
            pix_sum_sq = np.sum([pix_sum_sq, np.sum(np.square(input_np), axis=(1,2))], axis=0) 

        global_mean = pix_sum / total_pix_count
        global_var = (pix_sum_sq / total_pix_count) - (np.square(global_mean))
        global_std = np.sqrt(global_var)

        return global_mean, global_std

if __name__ == "__main__":
    my_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    og_ds = CIFAR10("/data", train=True,transform=my_transforms, target_transform=None, download=True)
    print(len(og_ds))
    settings = dict(experiment_ratio=20)
    imbalancer = TrainClassImbalancer(dataset=og_ds, extra_settings=settings)
    chosen_subset = imbalancer.get_chosen_subset()
    unchosen_subset = imbalancer.get_unchosen_subset()
    print(chosen_subset[3])
    print(unchosen_subset[1])
    print(len(chosen_subset))
    print(len(unchosen_subset))

    # print("Num target classes", np.unique(np.asarray(unchosen_subset[:, -1])))
    count = len(unchosen_subset)
    targets = np.zeros(count)
    for i in range(count):
        targets[i] = unchosen_subset[i][1] # type: ignore

    print(np.unique(targets))