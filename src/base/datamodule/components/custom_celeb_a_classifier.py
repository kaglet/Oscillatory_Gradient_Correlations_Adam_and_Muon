from typing import Any, Tuple
import logging
import skimage
import numpy as np
import os
from glob import glob
import torch
from torchvision.transforms import Resize
import torchvision
from torch.utils.data import DataLoader, Dataset
import pandas as pd

logger = logging.getLogger(__name__)

class CelebADataset(Dataset):
    def __init__(self, root:str='train', image_loader:Any=torchvision.io.read_image, transform:Any=None) -> None:
        super().__init__()
        imgs_path = "./data/CelebA/img_align_celeba/"
        all_imgs_path = os.path.join(imgs_path, '*.jpg')
        self.sorted_imgs_paths = sorted(os.listdir(all_imgs_path))
        n_imgs = len(self.sorted_imgs_paths)

        logger.info("Retrieved images!")

        attr_path = "./data/CelebA/list_attr_celeba.txt"
        attr_f = open(attr_path, 'r')
        self._num_imgs = int(attr_f.readline())
        header = attr_f.readline().split()

        partitions_path = "./data/CelebA/list_eval_partition.txt"
        partitions_df = pd.read_csv(partitions_path, delimiter=" ", names=["split_num"], usecols=[1])
        attr_df = pd.read_csv(attr_path, delimiter=" ", skiprows=2, names=header, usecols=["Male", "Blond_Hair"])
        attr_df.insert(0, "image_path", self.sorted_imgs_paths)
        attr_df[["Male", "Blond_Hair"]].replace(-1, 0, inplace=True)
        id_df = pd.read_csv(attr_path, delimiter=" ", skiprows=1, names=["image_id"], usecols=[0])
        self._final_df = pd.concat([attr_df, id_df, partitions_df], axis=0)
        self._final_df[0, ["image_path"]].astype(int)

        assert self._num_imgs == self._final_df.size, "Size mismatch, num images and dataframe"

        self._loader = image_loader
        self.transform = transform
        logger.info("Obtained imgs, feats, targets")


    def __len__(self):
        return self._num_imgs 

    def __getitem__(self, index):
        path = self._final_df[index, "image_path"].map(str)
        feats = self._final_df[index, "Male"]
        y = self._final_df[index, "Blond_Hair"]
        image = self._loader(path=str(path[0]))

        if self.transform is not None:
            image = self.transform(image)
        
        return image, feats, y

def get_preset_splits(imgs, features, targets, partitions_path: str = "./data/CelebA/list_eval_partition.txt"):
    """ 
    * Obtain preset official splits. 
    * To change ratios concatenate the train and val. 
    * DO NOT alter the test split. 
    """
    f_partitions = open(partitions_path, 'r')
    text = f_partitions.readlines()
    text = np.array(text)
    split_idx_assignments = text[:, 1]
    assert len(split_idx_assignments) == imgs.size(dim=0), "CelebA: Index assignment length does not match input images"
    logger.info("Read partitions file with size", len(text))

    train_imgs, train_feats, train_targets = [(img, feature, target) for img, feature, target, split_idx in zip(imgs, features, targets, split_idx_assignments) if split_idx == 0]
    val_imgs, val_feats, val_targets = [(img, feature, target) for img, feature, target, split_idx in zip(imgs, features, targets, split_idx_assignments) if split_idx == 1]
    test_imgs, test_feats, test_targets = [(img, feature, target) for img, feature, target, split_idx in zip(imgs, features, targets, split_idx_assignments) if split_idx == 2]    
    logger.info("Created splits", len(text))

    splits = dict({
        "train": {
            "imgs": train_imgs,
            "feats": train_feats,
            "targets": train_targets
        },
        "val": {
            "imgs": val_imgs,
            "feats": val_feats,
            "targets": val_targets
        },
        "test": {
            "imgs": test_imgs,
            "feats": test_feats,
            "targets": test_targets
        }
    })

    return splits 

if __name__ == "__main__":
    logging.basicConfig(filename='custom_celeb_a.log', level=logging.INFO)
    imgs, feats, targets = extract_data()
    splits = get_preset_splits(imgs, feats, targets)
    logger.info("Splits obtained")

    train_lens = len(splits["train"]["imgs"]), len(splits["train"]["feats"]), len(splits["train"]["targets"])
    logger.info("Train has len(s) for imgs, feats, targets: ", train_lens)

    val_lens = len(splits["val"]["imgs"]), len(splits["val"]["feats"]), len(splits["val"]["targets"])
    logger.info("Val has len(s) for imgs, feats, targets: ", val_lens)

    test_lens = len(splits["test"]["imgs"]), len(splits["test"]["feats"]), len(splits["test"]["targets"])
    logger.info("Test has len(s) for imgs, feats, targets: ", test_lens)