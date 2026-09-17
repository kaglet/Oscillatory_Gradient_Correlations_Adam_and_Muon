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
    def __init__(self, weighted_ratios: Dict[int, int], targets: np.ndarray):
        self.weighted_ratios = weighted_ratios
        self.targets = targets

    def fit_resample(self) -> np.ndarray:
        """ 
        - Randomly samples top R of each class where R is the preset ratio of the label.
        - Outputs all_idxs from self.targets to use to form new subset distribution.
        """
        targets = self.targets
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
        return all_idxs
    
if __name__ == "__main__":
    pass