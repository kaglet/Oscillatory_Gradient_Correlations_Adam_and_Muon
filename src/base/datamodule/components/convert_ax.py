import torch 
import numpy as np

def convert_img_axes(img: torch.Tensor | np.ndarray, channel_first=False) -> torch.Tensor:
    if type(img) is torch.Tensor:
        arr = (1,2,0) if img.shape[0] == 3 else (2,0,1)
        img = img.permute(arr)
    elif type(img) is np.ndarray:
        img = np.moveaxis(img, 0, -1) 

    return torch.as_tensor(img, dtype=torch.float32)