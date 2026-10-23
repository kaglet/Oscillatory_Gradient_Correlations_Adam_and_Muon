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
    def __init__(self, model: Module, params: list[list[dict[str, Any]]], optims: list[Optimizer], schedulers: list[LRScheduler], grad_accum_schedule: GradAccumSchedule = GradAccumSchedule({ 0: 1 }), num_classes: int = 10, labels: list[str] = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]):
        super().__init__(model, params, optims, schedulers, labels, grad_accum_schedule, num_classes)
        