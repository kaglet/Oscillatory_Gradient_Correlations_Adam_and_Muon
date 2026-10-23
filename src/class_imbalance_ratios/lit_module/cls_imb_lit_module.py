from typing import Any

from torch.nn import Module

from src.base.lit_module.components.custom_grad_accum import GradAccumSchedule
from src.base.lit_module.general_lit_module import DualOptLitModule

from torch.optim import Optimizer  
from torch.optim.lr_scheduler import LRScheduler

class ClsImbSLitModule(DualOptLitModule):
    """ 
    This assumes arguments for class imbalance to go with the DataModule used. 
    The default labels are CIFAR10. TODO: Change and make this generalized for 
    either CIFAR10 or CIFAR100, but a unified class like this is still useable I think.
    """
    def __init__(self, model: Module, params: list[list[dict[str, Any]]], optims: list[Optimizer], schedulers: list[LRScheduler], grad_accum_schedule: GradAccumSchedule = GradAccumSchedule({ 0: 1 }), num_classes: int = 10, labels: list[str] = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]):
        super().__init__(model, params, optims, schedulers, labels, grad_accum_schedule, num_classes)

class ClsImbLargeLitModule(DualOptLitModule):
    """ 
    This assumes arguments for class imbalance to go with the DataModule used. 
    The default labels are CIFAR10. TODO: Change and make this generalized for 
    either CIFAR10 or CIFAR100, but a unified class like this is still useable I think.
    """
    def __init__(self, model: Module, params: list[list[dict[str, Any]]], optims: list[Optimizer], schedulers: list[LRScheduler], labels: list[str], grad_accum_schedule: GradAccumSchedule = GradAccumSchedule({ 0: 1 }), num_classes: int = 10):
        super().__init__(model, params, optims, schedulers, labels, grad_accum_schedule, num_classes)
        