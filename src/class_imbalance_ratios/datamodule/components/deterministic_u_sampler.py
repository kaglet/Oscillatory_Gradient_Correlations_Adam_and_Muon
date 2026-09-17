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
        """ 
        With an input of targets array, two output masks can be retrieved:
        (1) -> chosen_mask: Matches to labels based off ratios
        (2) -> unchosen_mask: Unmatched labels. 
        This class does NOT shuffle. Targets must be shuffled beforehand. 
        The first R of a class are picked, where R is the ratio assigned.
        """
        self.weighted_ratios = weighted_ratios
        self.targets = targets

    def fit_resample(self) -> Tuple[np.ndarray, np.ndarray]:
        """ 
        - 
        """
        targets = self.targets
        if targets.ndim == 1:
            targets = targets.reshape((-1,1))

        chosen_mask = np.zeros_like(targets)

        for (lbl, R) in self.weighted_ratios.items():
            lbl_matcher = (targets == lbl)
            assert type(lbl_matcher) is np.ndarray, "lbl_matcher should be type np.ndarray"
            lbl_matcher[(lbl_matcher.cumsum()>R).argmax():] = 0
            np.sum((chosen_mask, lbl_matcher), out=chosen_mask, axis=0)

        unchosen_mask = ~chosen_mask
        return chosen_mask, unchosen_mask
    
if __name__ == "__main__":
    pass