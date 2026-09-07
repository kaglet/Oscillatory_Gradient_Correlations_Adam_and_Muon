import matplotlib.pyplot as plt 
import torch 
import numpy as np

from src.base.datamodule.components.convert_ax import convert_img_axes

__all__ = ["viz_img"]

def viz_img(img: torch.Tensor | np.ndarray, label:int = 1):
    img = convert_img_axes(img)
    plt.title('{}'.format(label))
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.show(block=True)