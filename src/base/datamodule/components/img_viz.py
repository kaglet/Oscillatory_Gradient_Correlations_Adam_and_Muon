import matplotlib.pyplot as plt 
import torch 
import numpy as np

from src.base.datamodule.components.convert_ax import convert_img_axes

__all__ = ["viz_img"]

def viz_img(img: torch.Tensor, label:int = 1):
    plt.title('{}'.format(label))
    plt.axis("off")
    plt.imshow(img.squeeze().to(dtype=torch.uint8), cmap="gray")
    plt.show(block=True)