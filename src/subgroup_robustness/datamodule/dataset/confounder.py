import os
from pathlib import Path
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from src.base.lit_module.components.models import model_attributes
from torch.utils.data import Dataset, Subset
from torchvision.transforms.functional import InterpolationMode

# Adaptation credit: @estija/CMID
class ConfounderDataset(Dataset):
    def __init__(self, root_dir: Path | str, target_name: str, confounder_names: list[str],
                 model_type: str, augment_data: bool):
        self._root_dir = root_dir
        self._target_name = target_name
        self._confounder_names = confounder_names
        self._augment_data = augment_data
        self._model_type = model_type

        # Read in attributes
        self._attrs_df = pd.read_csv(
            os.path.join(root_dir, 'data', 'list_attr_celeba.csv'))

        # Split out filenames and attribute names
        self._data_dir = os.path.join(self._root_dir, 'data', 'img_align_celeba')
        self._filename_array = self._attrs_df['image_id'].astype(dtype=np.string_).values
        self._attrs_df = self._attrs_df.drop(labels='image_id', axis='columns')
        self._attr_names = self._attrs_df.columns.copy()

        # Then cast attributes to numpy array and set them to 0 and 1
        # (originally, they're -1 and 1)
        self._attrs_np = np.asarray(self._attrs_df.values, dtype=np.float32)
        self._attrs_np[self._attrs_df == -1] = 0

        # Get the y values
        target_idx = self.attr_idx(self._target_name)
        self._y_array = self._attrs_np[:, target_idx]
        self._n_classes = 2

        # Map the confounder attributes to a number 0,...,2^|confounder_idx|-1
        self._confounder_idx = [self.attr_idx(a) for a in self._confounder_names]
        self._n_confounders = len(self._confounder_idx)
        confounders = self._attrs_np[:, self._confounder_idx]
        confounder_id = confounders @ np.power(2, np.arange(len(self._confounder_idx)))
        self._confounder_array = confounder_id

        # Map to groups
        self._n_groups = self._n_classes * pow(2, len(self._confounder_idx))
        self._group_array = (self._y_array*(self._n_groups/2) + self._confounder_array).astype('int')

        # Read in train/val/test splits 
        # From partition IDs
        self._split_df = pd.read_csv(
            os.path.join(root_dir, 'data', 'list_eval_partition.csv'))
        self._split_array = self._split_df['partition'].values
        self._split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        if model_attributes[self._model_type]['feature_type']=='precomputed':
            self._features_mat = torch.from_numpy(np.load(
                os.path.join(root_dir, 'features', model_attributes[self._model_type]['feature_filename']))).float()
            self._train_transform = None
            self._eval_transform = None
        else:
            self._features_mat = None
            self._train_transform = get_transform_celebA(self._model_type, train=True, augment_data=augment_data)
            self._eval_transform = get_transform_celebA(self._model_type, train=False, augment_data=augment_data)

    def attr_idx(self, attr_name) -> int:
        idx = self._attr_names.get_loc(attr_name)
        assert type(idx) is int, f"Attribute index of {type(idx)} should be int" 
        return idx

    def __len__(self):
        return len(self._filename_array)

    def __getitem__(self, idx):
        y = self._y_array[idx]
        g = self._group_array[idx]

        if model_attributes[self._model_type]['feature_type']=='precomputed':
            assert self._features_mat is not None, "features_mat is type None, when Tensor expected"
            x = self._features_mat[idx, :]
        else:
            img_filename = os.path.join(
                self._data_dir,
                str(self._filename_array[idx]))
            img = Image.open(img_filename).convert('RGB')
            # Figure out split and transform accordingly
            if self._split_array[idx] == self._split_dict['train'] and self._train_transform:
                img = self._train_transform(img)
            elif (self._split_array[idx] in [self._split_dict['val'], self._split_dict['test']] and
              self._eval_transform):
                img = self._eval_transform(img)
            # Flatten if needed
            if model_attributes[self._model_type]['flatten']:
                assert type(img) is torch.Tensor
                assert len(img.shape) == 3
                img = img.view(-1)
            x = img

        return x,y,g

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            assert split in ('train','val','test'), split+' is not a valid split'
            mask = self._split_array == self._split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac<1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, list(indices))
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self._n_groups/self._n_classes)
        c = group_idx % (self._n_groups//self._n_classes)

        group_name = f'{self._target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self._n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self._confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name

def get_transform_celebA(model_type, train, augment_data):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    if model_attributes[model_type]['target_resolution'] is not None:
        target_resolution = model_attributes[model_type]['target_resolution']
    else:
        target_resolution = (orig_w, orig_h)

    if (not train) or (not augment_data):
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Orig aspect ratio is 0.81, so we don't squish it in that direction any more
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform