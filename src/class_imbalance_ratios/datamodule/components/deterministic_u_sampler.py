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


    
if __name__ == "__main__":
    pass