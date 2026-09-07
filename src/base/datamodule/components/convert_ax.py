import torch 
import numpy as np

def convert_img_axes(img: torch.Tensor | np.ndarray) -> torch.Tensor:
    if img.shape[0] == 3 and type(img) is torch.Tensor:
        img = img.permute(1,2,0)
    elif img.shape[0] == 3 and type(img) is np.ndarray:
        img = np.moveaxis(img, 0, -1) 

    return torch.as_tensor(img, dtype=torch.float32)