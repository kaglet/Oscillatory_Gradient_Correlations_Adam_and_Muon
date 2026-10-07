import torch 
import numpy as np

def convert_img_axes(img: torch.Tensor | np.ndarray, channel_first=False) -> torch.Tensor:
    """
    Convert image to desired (C, H, W) or (H, W, C).
    Return if already satisfies requirements, but wrapped as tensor. 
    Accepts numpy or tensor value. 
    This on demand method prevents image confusion when you desire a consistent pipeline despite other APIs.   
    """
    
    if channel_first == False:
        if img.shape[2] == 3: return torch.as_tensor(img, dtype=torch.float32)

        if type(img) is torch.Tensor:
            arr = (1,2,0)
            img = img.permute(arr)
        elif type(img) is np.ndarray:
            img = np.moveaxis(img, 0, -1) 
    elif channel_first ==  True: 
        if img.shape[0] == 3: return torch.as_tensor(img, dtype=torch.float32)
        
        if type(img) is torch.Tensor:
            arr = (2,0,1) 
            img = img.permute(arr)
        elif type(img) is np.ndarray:
            img = np.moveaxis(img, -1, 0) 

    return torch.as_tensor(img, dtype=torch.float32)