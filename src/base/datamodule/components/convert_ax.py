import torch 
import numpy as np

def convert_img_axes(img: torch.Tensor | np.ndarray, channel_first=False, num_channels=3) -> torch.Tensor | np.ndarray:
    """
    Convert image to desired (C, H, W) or (H, W, C).
    Return if already satisfies requirements, but wrapped as native. 
    Accepts numpy or tensor value. 
    This on demand method prevents image confusion 
    when you desire a consistent pipeline despite other APIs.   
    """
    
    if channel_first == False:
        if img.shape[2] == num_channels: return img

        if type(img) is torch.Tensor:
            arr = (1,2,0)
            img = img.permute(arr)
        elif type(img) is np.ndarray:
            img = np.moveaxis(img, 0, -1) 
    elif channel_first ==  True: 
        if img.shape[0] == num_channels: return img
        
        if type(img) is torch.Tensor:
            arr = (2,0,1) 
            img = img.permute(arr)
        elif type(img) is np.ndarray:
            img = np.moveaxis(img, -1, 0) 

    return img