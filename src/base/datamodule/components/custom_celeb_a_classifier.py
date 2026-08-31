
from __future__ import division, print_function, absolute_import
from typing import Any, Tuple

from skimage import color, io
import skimage
import numpy as np
import os
from glob import glob

import math

def extract_data() -> Tuple[Any, Any, Any]:
    files_path = "./data/CelebA/img_align_celeba/"

    glasses_files_path = os.path.join(files_path, '*.jpg')

    glasses_files = sorted(glob(glasses_files_path))

    n_files = len(glasses_files)
    size_image1=27
    size_image2=33
    all_input_imgs = np.zeros((n_files, size_image1, size_image2, 3), dtype='float32')
    all_y = np.zeros(n_files)
    count = 0
    for f in glasses_files:
        try:
            img = io.imread(f)
            new_img = skimage.transform.resize(img, (27, 33, 3))
            all_input_imgs[count] = np.array(new_img)
            all_y[count] = 0
            count += 1
        except:
            continue
    attr_target=[]
    attr_feature=[]
    g = open('./list_attr_celeba.txt', 'r')
    text = g.readlines()
    text = np.array(text)
    attr2idx = dict()
    for i, attr in enumerate(text[1].split()):
        attr2idx[attr] = i
    attr_index_target = attr2idx['Blond_Hair']#'Eyeglasses'
    attr_index_feature = attr2idx['Male']#'Eyeglasses'

    for i in text[2:]:
        value = i.split()
        attr_target.append(value[attr_index_target + 1]) #First index is image name
        attr_feature.append(value[attr_index_feature + 1]) #First index is image name

    attr_target = np.array(attr_target,dtype= np.float32)
    attr_feature = np.array(attr_feature,dtype= np.float32)

    #print("Converting Label.................")
    for i in range(0,len(attr_target)):
        if (attr_target[i] == 1):
            all_y[i]=1
        else:
            all_y[i]=0
    all_y = np.array(all_y)

    # Return image, other_feats, targets
    return all_input_imgs, attr_feature, all_y

def get_splits():
    g = open('./list_eval_partition.txt', 'r')
    text = g.readlines()
    text = np.array(text)
    split_idx_assignments = text[:, 1]
    # TODO: Easy filter. 