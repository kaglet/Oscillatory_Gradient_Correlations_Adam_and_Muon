from torchvision.models import resnet18
from torch import nn
from typing import Any

# TODO: Edit type checked of resnet in fact a callable/mapping
def create_resnet_model(net: Any = resnet18, num_classes: int = 10, num_in: int=3, num_out: int = 64, kern_size: tuple[int, int] = (3,3), stride: tuple[int, int] = (1,1), padding: tuple[int, int] = (1,1), bias: bool = False) -> nn.Module:
    """ 
    Architecture dependent function to create and minimally edit a given resnet model. 
    """
    model = net(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=kern_size, stride=stride, padding=padding, bias=bias)
    model.maxpool = nn.Identity()
    return model