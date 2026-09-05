import math
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
from torchvision import transforms
from torchvision.transforms import v2

logger = logging.getLogger(__name__)

class CelebADataset(Dataset):
    def __init__(self, root:str='train', image_loader:Any=torchvision.io.decode_image, transform:Any=None) -> None:
        super().__init__()
        imgs_path = "./data/CelebA/img_align_celeba/"
        all_imgs_path = [imgs_path + local_path for local_path in os.listdir(imgs_path)]  
        self._sorted_imgs_paths = sorted(all_imgs_path)

        attr_path = "./data/CelebA/list_attr_celeba.txt"
        attr_f = open(attr_path, 'r')
        self._num_imgs = int(attr_f.readline())
        header = attr_f.readline().split()

        partitions_path = "./data/CelebA/list_eval_partition.txt"
        partitions_df = pd.read_csv(partitions_path, sep=r' |  ', engine='python', names=["split_num"], usecols=[1])
        attr_df = pd.read_csv(attr_path, sep=r' |  ', engine='python', skiprows=2, names=["male", "blond_hair"], usecols=[header.index("Male"), header.index("Blond_Hair")], dtype=pd.Int64Dtype)
        attr_df.replace(-1, 0, inplace=True)
        attr_df["image_path"] = self._sorted_imgs_paths

        id_df = pd.read_csv(attr_path, sep=r' |  ', engine='python', skiprows=2, names=["image_id"], usecols=[header.index("image_id")])

        print(attr_df.info(), partitions_df.info(), id_df.info())
        self._final_df = pd.concat([attr_df, id_df, partitions_df], axis=1)

        logger.info(self._final_df.info())
        assert self._num_imgs == self._final_df.shape[0], f"Size mismatch, num images and dataframe {self._num_imgs} vs {self._final_df.size}"

        self._loader = image_loader
        self._transforms = transform

        paths = self._final_df.loc[:, "image_path"]
        feats = torch.tensor(self._final_df.loc[:, "male"]) 
        y = torch.tensor(self._final_df.loc[:, "blond_hair"]) 
        self._paths = paths
        self._targets = y 
        self._feats = feats
        
    def __len__(self):
        return self._num_imgs 

    def __getitem__(self, index):
        """ 
        Returns tuple of image, single spurious feature for experiment, target
        """
        # Return image as transformed tensor, on demand 
        image = self._transforms(self._loader(self._paths.loc[index]))
        # TODO: Apply target transforms if necessary, what you do to input space, output space needs e.g. To Tensor as in CIFAR10 dataset
        return image, self._feats[index].unsqueeze(0), self._targets[index].unsqueeze(0)

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
    dataset = CelebADataset(transform=celeb_a_transforms)
    first_data = dataset[0]
    img, feats, target = first_data

    batch_size = 16
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8, persistent_workers=True)
    dataiter = iter(dataloader)

    n_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples/batch_size)
    print(total_samples, n_iterations) 

    for epoch in range(n_epochs):
        for i, (img, feats, y) in enumerate(dataloader):
            if (i+1) % 5 == 0:
                print(f"epoch {epoch+1}/{n_epochs}, step {i+1}/{n_iterations}, imgs: {img.shape}, feats: {feats.shape}, targets: {y.shape}")