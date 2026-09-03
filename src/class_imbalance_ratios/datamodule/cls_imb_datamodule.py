import logging
import math
import os
import random
from typing import Any, Callable, Dict, Literal, Optional, Type
from collections import Counter
import torch
import lightning as L
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from torch.utils.data import random_split, DataLoader, TensorDataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import v2

from imblearn.under_sampling import RandomUnderSampler

from src.base.datamodule.cifar_datamodule import CIFARDataModule

__all__ = ["ClsImbDataModule"]

class ClsImbDataModule(CIFARDataModule):
    # TODO" Unify the data directory from env see old template
    def __init__(
        self,
        weighted_ratios: Dict[int, float] | None = None,
        downsampler_ratio: int | None = None,
        *args: Any, 
        **kwargs: Any
    ):
        """
        * For the experiment set a ratio of R, where R % num_cifar_classes = 0
        * Half of the classes of CIFAR are set to the upperbound. The rest are divided by the ratio.
        * - weighted_ratios: Dict[int, float] accepts dynamic weights of length equivalent to CIFAR10 classes or CIFAR100 classes depending on the choice of kw arg. Weights will be normalised by proportion of total.
        * - downsampler_ratio: int, accepts a positive number, to apply strict logic of random 50% of classes chosen as dataset for 1 run of training (usage of this datamodule instance)
        * Exclusively one of the above two must be set, if both are set, dict input is discarded by convention. 
        """
        super().__init__(*args, **kwargs)
        if downsampler_ratio is not None and weighted_ratios is not None: 
            weighted_ratios = None 

        self._target_weights = weighted_ratios
        self._downsampler_ratio = downsampler_ratio 

    def setup(self, stage: str)  -> None:
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            cifar_full = self.ds(self.data_dir, train=True, transform=self.train_transforms)
            self._cifar_train, self._cifar_val = random_split(
                cifar_full, [0.8, 0.2]
            )

            sampler = RandomUnderSampler(sampling_strategy="dict")
            _, y_full = cifar_full
            sampler.sampling_strategy_ = self._get_ratios(y_full)

            x_train, y_train = self._cifar_train
            x_val, y_val = self._cifar_val

            new_samples = sampler.fit_resample(x_train, y_train)
            self._cifar_train = TensorDataset(torch.tensor(new_samples[0]), torch.tensor(new_samples[1]))
            new_samples = sampler.fit_resample(x_val, y_val)
            self._cifar_val = TensorDataset(torch.tensor(new_samples[0]), torch.tensor(new_samples[1]))

        # Ensure test is never mixed up with train, val for any model, and is seen once and once only at the end
        if stage == "test":
            self._cifar_test = self.ds(self.data_dir, train=False, transform=self.test_transforms)

        if stage == "predict":
            self._cifar_predict = self.ds(self.data_dir, train=False, transform=self.test_transforms)

    def _get_ratios(self, targets) -> Dict[int, int]:
        """ 
        Outputs normalised weights dictionary from single down sampling ratio xor weight dict.
        If not assignable outputs with appropriate assertion errors violating assumptions. 
        """
        labels = Counter(targets).keys()
        len_ds = len(self._cifar_train)
        len_labels = len(labels)
        final_weights = None

        if self._target_weights is not None:
            total = sum([val for val in self._target_weights.values()])
            assert total == len_labels, f"Dict given has len {len_labels} != target domain len {len_ds}"
            final_weights = {i:math.ceil((r/total)*len_ds) for (i,r) in self._target_weights.items()}
        elif self._downsampler_ratio is not None:
            class_ratios = [(i,1) for i in range(len_ds)]
            random.shuffle(class_ratios)
            slice_first = class_ratios[math.ceil(len_ds / 2) : ]
            slice_last = class_ratios[ : math.ceil(len_ds / 2)]
            slice_last = [(i,self._downsampler_ratio) for (i,r) in slice_last]
            new_ratios = slice_first + slice_last
            final_weights = dict(new_ratios)

        assert final_weights is not None, "Final weights not assigned, aborting"

        return final_weights

# Entry point
if __name__ == "__main__":
    """
    Confirms smooth run of dataloader, native image sizes, with minimal transforms
    """
    logging.basicConfig(filename='custom_celeb_a.log', level=logging.INFO)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # ImageNet standard norm
                                    std=[0.229, 0.224, 0.225])
    # TODO: Resize images smaller and train on subset for efficiency and MVP.
    celeb_a_transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        normalize
    ])
    
    dmodule = ClsImbDataModule(downsampler_ratio=20) 
    dmodule.setup(stage="fit")
    dataloader = dmodule.train_dataloader()
    dataiter = iter(dataloader)
    batch_size = 4
    n_epochs = 2
    total_samples = 50000
    n_iterations = math.ceil(total_samples/batch_size)
    print(total_samples, n_iterations) 

    for epoch in range(n_epochs):
        for i, (img, feats, y) in enumerate(dataloader):
            if (i+1) % 5 == 0:
                print(f"epoch {epoch+1}/{n_epochs}, step {i+1}/{n_iterations}, imgs: {img.shape}, feats: {feats.shape}, targets: {y.shape}") 

    # TODO: Do the same for stage = "val"