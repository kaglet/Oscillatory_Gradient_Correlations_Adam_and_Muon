import torch
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from torch import nn
from typing import Any, Type
from torchvision.models import Weights

__all__ = ["create_resnet_model"]

# TODO: Edit type checked of resnet in fact a callable/mapping
def create_resnet_model(net: Any, num_classes: int = 100, num_in: int=3, num_out: int = 64, kern_size: list[int] = [3,3], stride: list[int] = [1,1], padding: list[int] = [1,1], bias: bool = True, weights: Any = ResNet18_Weights.DEFAULT) -> nn.Module:
    """ 
    Architecture dependent function to create and minimally edit a given resnet model. 
    ImageNet finetuning is used from the research paper. Architecture alterations are not stated for random init case, which should not be the case. 
    """
    kern_size_n = (kern_size[0], kern_size[1])
    stride_n = stride[0], stride[1]
    padding_n = padding[0], padding[1]

    model = net()
    model.conv1 = nn.Conv2d(num_in, num_out, kernel_size=kern_size_n, stride=stride_n, padding=padding_n, bias=bias)
    model.maxpool = nn.Identity()
    fc_num_in = model.fc.in_features 
    model.fc = nn.Linear(fc_num_in, num_classes) 
    
    return model

if __name__ == "__main__":

    def print_metadata(model: torch.nn.Module):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters    : {total:,}")
        print(f"Trainable parameters: {trainable:,}  ({100*trainable/total:.2f}%)")
        for module in model.fc.modules(): # type: ignore
            print(module)

    og_model = resnet18()
    new_model = create_resnet_model(resnet18)
    print_metadata(og_model)
    print_metadata(new_model)