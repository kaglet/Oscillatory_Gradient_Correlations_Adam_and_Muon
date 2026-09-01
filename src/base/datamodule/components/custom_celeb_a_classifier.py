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
        self._sorted_imgs_paths = sorted(glob(all_imgs_path))
        n_imgs = len(self._sorted_imgs_paths)

        logger.info("Retrieved images!")

        attr_path = "./data/CelebA/list_attr_celeba.txt"
        attr_f = open(attr_path, 'r')
        self._num_imgs = int(attr_f.readline())
        header = attr_f.readline().split()
        print("Header is: ", header)
        print("Male idx", header.index("Male"), "Blond_Hair idx: ", header.index("Blond_Hair"))
        partitions_path = "./data/CelebA/list_eval_partition.txt"
        partitions_df = pd.read_csv(partitions_path, sep=r' |  ', engine='python', names=["split_num"], usecols=[1])
        attr_df = pd.read_csv(attr_path, sep=r' |  ', engine='python', skiprows=2, names=["male", "blond_hair"], usecols=[header.index("Male"), header.index("Blond_Hair")], dtype=pd.Int64Dtype)
        attr_df.insert(loc=0, column="image_path", value=self._sorted_imgs_paths)
        attr_df.replace(-1, 0, inplace=True)

        id_df = pd.read_csv(attr_path, sep=r' |  ', engine='python', skiprows=2, names=["image_id"], usecols=[header.index("image_id")])

        print(attr_df.info(), partitions_df.info(), id_df.info())
        self._final_df = pd.concat([attr_df, id_df, partitions_df], axis=1)

        print(self._final_df.info())
        assert self._num_imgs == self._final_df.shape[0], f"Size mismatch, num images and dataframe {self._num_imgs} vs {self._final_df.size}"

        self._loader = image_loader
        self.transform = transform
        logger.info("Obtained imgs, feats, targets")


    def __len__(self):
        return self._num_imgs 

    def __getitem__(self, index):
        """ 
        Returns tuple of image, single spurious feature for experiment, target
        """
        path = self._final_df[index, "image_path"].map(str)
        feats = torch.tensor(self._final_df[index, ["Male"]].values) 
        y = torch.tensor(self._final_df[index, ["Blond_Hair"]].values) 
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
    dataset = CelebADataset()
    first_data = dataset[0]
    img, feats, target = first_data
    print(img, feats, target)