from typing import Any

from torchvision.datasets import CIFAR10, CIFAR100
import torch 
from torch.utils.data import Dataset

class TupledSubset(Dataset):
    r"""
    Subset of a dataset at specified indices. Unlike the ordinary `Subset` this does not lose the input data, targets distinction. 
    It is standardized to behave much like a regular Dataset when utilized. 
    A `Subset` can access on demand the data, target one at a time with `get_item`. 
    This can access both .data and .targets.
    By Subclassing `Dataset` and providing a `get_item` like Subset, this can be passed to a data loader as usual. 

    @ Credit: Adapted from @kowshik_thopalli in PyTorch discussions.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """
    def __init__(self, dataset: CIFAR10 | CIFAR100, indices: list[int], targets: torch.Tensor | Any):
        self.dataset = dataset
        self.indices = indices
        targets_hold = torch.ones(len(dataset)).type(torch.long) *300 #( some number not present in the #labels just to make sure
        targets_hold[self.indices] = targets 
        self.targets = targets_hold
    def __getitem__(self, idx):
        image = self.dataset[self.indices[idx]][0]
        label = self.targets[self.indices[idx]]
        return (image, label)

    def __len__(self):
        return len(self.indices)