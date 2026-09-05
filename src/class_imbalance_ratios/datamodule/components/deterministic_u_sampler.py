__all__ = ["DeterministicUnderSampler"]

import math
from typing import Dict, Tuple
import numpy as np
from numpy import ndarray
import torch
from torch.utils.data import Dataset, TensorDataset

class DeterministicUnderSampler():
    def __init__(self, weighted_ratios: Dict[int, int], data: ndarray, targets: ndarray):
        """
        Expects dtype set on inputs, preferrably float32 for ML training.
        """
        self.weighted_ratios = weighted_ratios
        self.data = data.squeeze()
        self.targets = targets.squeeze()

    def fit_resample(self) -> TensorDataset:
        """ 
        Stitches together subsets of data selected for a particular label and ratio combination.
        Returns a subset of the dataset based off the sampling ratios. 
        """
        assert(len(self.data) == len(self.targets)), "Length mismatch, data and targets"
        choices = np.random.choice(np.array(range(len(self.data))), size=len(self.data))
        data = self.data[choices]
        targets = self.targets[choices]
        subset_data = []
        subset_targets = []

        for (lbl, R) in self.weighted_ratios.items():
            lbl_matcher = (targets == lbl)
            # select only top R for targets and data
            target_matches = targets[lbl_matcher]
            top_r = target_matches[:R]
            subset_targets.append(top_r)

            data_matches = data[lbl_matcher]
            top_r = data_matches[:R, :]
            subset_data.append(top_r)

        subset_targets = torch.tensor(np.concatenate(subset_targets, axis=0))
        subset_data = torch.tensor(np.concatenate(subset_data, axis=0))

        return TensorDataset(subset_data, subset_targets)
    
if __name__ == "__main__":
    pass