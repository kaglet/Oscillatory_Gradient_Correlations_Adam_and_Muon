import os
from pathlib import Path
import PIL
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from src.base.lit_module.components.models import model_attributes
from torch.utils.data import Dataset, Subset
from src.subgroup_robustness.datamodule.dataset.confounder import ConfounderDataset

class CelebADataset(ConfounderDataset):
    """
    CelebA dataset (already cropped and centered).
    Note: idx and filenames are off by one.
    """

    def __init__(self, root_dir: Path | str, target_name: str, confounder_names: list[str],
                 model_type: str, augment_data: bool):
        super().__init__(root_dir=root_dir, target_name=target_name, confounder_names=confounder_names, model_type=model_type, augment_data=augment_data)


