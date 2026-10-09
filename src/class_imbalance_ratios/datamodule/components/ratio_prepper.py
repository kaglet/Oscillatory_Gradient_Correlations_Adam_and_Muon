import math
import numpy as np
import torch

class CountsPrepperFromRatio():
    def __init__(self, train_map: dict[int, int], val_map: dict[int, int], test_map: dict[int, int], downsampler_ratio: int, slice_random: bool = True) -> None:
        train_map = train_map
        train_counts = np.asarray(list(train_map.values()))
        train_labels = np.asarray(list(train_map.keys()))
        if slice_random:
            downsample_idxs = np.random.choice(np.arange(train_counts.size), size=math.floor(train_counts.size / 2), replace=False)
        else:
            downsample_idxs = np.arange(train_counts.size)[:5]

        train_counts[downsample_idxs] = train_counts[downsample_idxs] / downsampler_ratio

        self.train_map: dict[int, int] = {int(l): int(c) for l, c in zip(train_labels, train_counts)}

        self.val_map: dict[int, int] = val_map

        self.test_map: dict[int, int] = test_map

class ReshuffleCountMap():
    """
    From original map get a remap with shuffled values. 
    Consistent  input -> output types prevents confusing type conversions. 
    This is stochastic shuffling dictated by the torch manual seed. 

    """
    def __init__(self, counts_map: dict[int, int]) -> None:
        self._counts_map = counts_map
        counts = torch.as_tensor(list(counts_map.values()))
        idxs = torch.randperm(counts.shape[0]) # torch lacks simple np.shuffle like function. 
        self._new_counts = counts[idxs]

    @property
    def remapped(self) -> dict[int, int]:
        new_counts_map = {k:int(v) for k,v in zip(self._counts_map.keys(), self._new_counts)}
        return new_counts_map

if __name__ == "__main__":
    num_classes = 10
    labels = list(range(num_classes))

    train_class_counts = [3500 for _ in range(num_classes)]
    train_class_counts = [3500 for _ in range(num_classes)]

    train_map = {k:v for k,v in zip(labels, train_class_counts)}

    val_class_counts = [1000 for _ in range(num_classes)]
    val_map = {k:v for k,v in zip(labels, val_class_counts)}

    test_class_counts = [500 for _ in range(num_classes)]
    test_map = {k:v for k,v in zip(labels, test_class_counts)}

    ReshuffleCountMap(train_map)
    prepper = CountsPrepperFromRatio(downsampler_ratio=20, train_map=train_map, val_map=val_map, test_map=test_map)
    print(prepper.train_map)

