__all__ = ["DeterministicUnderSampler"]

import math
from typing import Any, Dict, Tuple
import numpy as np
from numpy import array
import torch
from torch.utils.data import Dataset, TensorDataset

from src.base.datamodule.components.cifar_10_helper import CIFAR10Helper
from src.base.datamodule.components.img_viz import viz_img

class DeterministicUnderSampler():
    def __init__(self, weighted_ratios: Dict[int, int], data: Any, targets: Any):
        """
        Expects dtype set on inputs, preferrably float32 for ML training.
        """
        self.weighted_ratios = weighted_ratios
        self.data = data
        self.targets = targets

    def fit_resample(self) -> Tuple[np.ndarray, np.ndarray]:
        """ 
        Stitches together subsets of data selected for a particular label and ratio combination.
        Returns a subset of the dataset based off the sampling ratios. 
        """
        assert(len(self.data) == len(self.targets)), "Length mismatch, data and targets"
        choices = np.random.choice(np.array(range(len(self.data))), size=len(self.data), replace=False)
        targets = self.targets[choices]
        if targets.ndim == 1:
            targets = targets.reshape((-1,1))

        all_idxs = []

        for (lbl, R) in self.weighted_ratios.items():
            lbl_matcher = (targets == lbl)
            idxs_matched = np.nonzero(lbl_matcher)
            subset_idxs = np.random.choice(idxs_matched, R)
            # select only top R for targets and data
            all_idxs.append(subset_idxs)

        all_idxs = np.concatenate(all_idxs, axis=0)
        subset_data = self.data[all_idxs]
        subset_targets = self.targets[all_idxs]
        return subset_data, subset_targets
    
if __name__ == "__main__":
    pass