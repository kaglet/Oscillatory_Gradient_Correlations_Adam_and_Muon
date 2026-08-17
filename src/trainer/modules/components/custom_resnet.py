from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from torch import nn
from typing import Any, Type
from torchvision.models import Weights

__all__ = ["create_resnet_model"]

# TODO: Edit type checked of resnet in fact a callable/mapping
def create_resnet_model(net: Any = resnet18, num_classes: int = 1000, num_in: int=3, num_out: int = 64, kern_size: tuple[int, int] = (3,3), stride: tuple[int, int] = (1,1), padding: tuple[int, int] = (1,1), bias: bool = False, weights: Any = ResNet18_Weights.DEFAULT) -> nn.Module:
    """ 
    Architecture dependent function to create and minimally edit a given resnet model. 
    """
    model = net(weights=weights, num_classes=num_classes)
    model.conv1 = nn.Conv2d(num_in, num_out, kernel_size=kern_size, stride=stride, padding=padding, bias=bias)
    model.maxpool = nn.Identity()
    return model