from typing import Any, Tuple
import logging
import skimage
import numpy as np
import os
from glob import glob
import torch
from torchvision.transforms import Resize
import torchvision

logger = logging.getLogger(__name__)

# Adapted from @Veera93
def extract_data(attr_path: str = "./data/CelebA/list_attr_celeba.txt", imgs_path: str = "./data/CelebA/img_align_celeba/") -> Tuple[Any, Any, Any]:
    all_imgs_path = os.path.join(imgs_path, '*.jpg')
    logger.info("Retrieved images!")
    sorted_imgs_path = sorted(glob(all_imgs_path))

    n_files = len(sorted_imgs_path)
    size_image1=27
    size_image2=33
    all_input_imgs = torch.zeros((n_files, size_image1, size_image2, 3), dtype=torch.float32)
    all_y = np.zeros(n_files)
    count = 0
    for f in sorted_imgs_path:
        try:
            img = torchvision.io.read_image(f)
            all_input_imgs[count] = img.detach().clone()
            logger.info("Transformed image of count: ", count)

            all_y[count] = 0
            count += 1
        except:
            continue
    logger.info("Transformed images with count: ", count)

    attr_target=[]
    attr_feature=[]
    f_attributes = open(attr_path, 'r')
    text = f_attributes.readlines()
    text = np.array(text)

    logger.info("Obtained attributes file and stored as array with size:", len(text))

    attr2idx = dict()
    for i, attr in enumerate(text[1].split()):
        attr2idx[attr] = i
    attr_index_target = attr2idx['Blond_Hair']
    attr_index_feature = attr2idx['Male']

    for i in text[2:]:
        value = i.split()
        attr_target.append(value[attr_index_target + 1]) #First index is image name
        attr_feature.append(value[attr_index_feature + 1]) #First index is image name

    attr_target = torch.tensor(attr_target, dtype = torch.float32)
    attr_feature = torch.tensor(attr_feature, dtype = torch.float32)

    for i in range(0,len(attr_target)):
        if (attr_target[i] == 1):
            all_y[i]=1
        else:
            all_y[i]=0
    all_y = torch.tensor(all_y, dtype = torch.float32)
    logger.info("Obtained imgs, feats, targets")

    # Return image, other_feats, targets
    return all_input_imgs, attr_feature, all_y

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