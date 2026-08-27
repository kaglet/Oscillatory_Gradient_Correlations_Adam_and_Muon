
"""
Anything configurable or static in a configurable aspect is defined here and assembled later.
"""

import os
import multiprocessing
from typing import Any, Type

from omegaconf import OmegaConf
import torchvision
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from hydra_zen import instantiate, just, launch, make_custom_builds_fn, make_config, make_custom_builds_fn, zen, store
from muon import MuonWithAuxAdam
from torch.optim import Muon, Adam
from torch.optim.lr_scheduler import LRScheduler
from train_and_val import train_and_val
from trainer.dataloader.cifar_dataloaders import CIFAR100DataModule, CIFAR10DataModule
from trainer.dataloader.components.mnist_normalization import mnist_normalization
from trainer.dataloader.mnist_dataloader import MNISTDataModule, SimpleMNISTDataModule
from trainer.modules.components.custom_grad_accum import GradAccumSchedule
from trainer.modules.general_lit_module import DualOptLitModule 
from trainer.modules.components.custom_resnet import create_resnet_model
from trainer.modules.components.simple_net import SimpleCNN
from trainer.per_optim_params.resnet_params.get_params import get_Adam_params, get_dist_Muon_Adam_params, get_dual_opt_params
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from hydra_zen.third_party.beartype import validates_with_beartype
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization

builds = make_custom_builds_fn(populate_full_signature=True)
pbuilds = make_custom_builds_fn(
    zen_partial=True, 
    populate_full_signature=True,
    zen_wrappers=validates_with_beartype
)

# ---- LIGHTNING MODULE CONFIGS BEGIN ---- #
# Define base params cfg
dual_optim_params_cfg = pbuilds(
    get_dual_opt_params,
)

single_optim_params_cfg = pbuilds(
    get_Adam_params,
)

# Define base vision model cfg
resnet18_model_cfg = builds(create_resnet_model, net=just(resnet18))
resnet50_model_cfg = builds(create_resnet_model, net=just(resnet50))

cls_imb_s_num_classes = 10
cls_imb_l_num_classes = 100
sbg_rob_num_classes = 5
sp_pix_num_classes = 10

cls_imb_model_s_cfg = resnet18_model_cfg(num_classes=cls_imb_s_num_classes, num_in=3)
cls_imb_model_l_cfg = resnet50_model_cfg(num_classes=cls_imb_l_num_classes, num_in=3) 
sbg_rob_model_cfg = resnet50_model_cfg(num_classes=sbg_rob_num_classes, num_in=3)
sp_pix_model_cfg = builds(SimpleCNN, num_classes=sp_pix_num_classes, num_in=1)

# Optimiser variations
# Default in distributed
mu_aux_adam_pcfg = [pbuilds(MuonWithAuxAdam)]
# Custom for non-distributed
dual_optims_pcfg = [pbuilds(Muon), pbuilds(Adam)]
# single optimiser training  
single_optim_pcfg = [pbuilds(Adam)]

# Gradient accum schedules 
grad_accum_schedule_cfg = builds(GradAccumSchedule, {0: 32})

# LR Schedulers
lr_sch_pcfg = [pbuilds(LRScheduler), pbuilds(LRScheduler)]

# Lighning Module variations 
cls_imb_s_module_cfg = pbuilds(
    DualOptLitModule, 
    grad_accum_schedule = grad_accum_schedule_cfg,
    num_classes = cls_imb_s_num_classes
)

cls_imb_l_module_cfg = pbuilds(
    DualOptLitModule, 
    grad_accum_schedule = grad_accum_schedule_cfg,
    num_classes = cls_imb_l_num_classes
)

sp_pix_module_cfg = pbuilds(
    DualOptLitModule, 
    grad_accum_schedule = grad_accum_schedule_cfg,
    num_classes = sp_pix_num_classes
)

sbg_rob_module_cfg = pbuilds(
    DualOptLitModule, 
    grad_accum_schedule = grad_accum_schedule_cfg,
    num_classes = sbg_rob_num_classes
)

# ---- TRAINER CONFIGS START ---- #
# TODO: Change val accuracy to be the best on on a subgroup not overall
max_checkp_cfg = builds(
    ModelCheckpoint,    
    save_top_k=1, 
    mode="max", 
    # TODO: As Unit 6.1 of PL docs do not cover ensuring consistency, find out how across names.
    monitor="val/acc", 
    save_last=True
)

chkpts = [max_checkp_cfg]

# Define base trainer cfg
trainer_cfg = builds(
    pl.Trainer, 
    max_epochs=4,
    deterministic=False,
    accelerator="gpu",
    benchmark=False,
    callbacks=max_checkp_cfg,
    default_root_dir=os.getcwd(),
    detect_anomaly=False,
    devices=1
    # precision = "bf16-mixed"
)  

# ---- TRAINER CONFIGS COMPLETE ---- #
# ---- DATAMODULE CONFIGS START ---- #
def create_transforms(transforms: torchvision.transforms.Compose) -> torchvision.transforms.Compose:
    """"
    Simplest workaround to define and configure lists as per Hydra-zen GitH discussion
    """
    return transforms

horizontal_flip = builds(torchvision.transforms.RandomHorizontalFlip)
vertical_flip = builds(torchvision.transforms.RandomVerticalFlip)
crops = builds(torchvision.transforms.RandomCrop, 20)
to_tensor = builds(torchvision.transforms.ToTensor)
cifar_10_norm = builds(cifar10_normalization)
cifar_100_norm = builds(imagenet_normalization)
mnist_norm = builds(mnist_normalization)

resnet18_transforms_cfg = builds(ResNet18_Weights.DEFAULT.transforms)
resnet50_transforms_cfg = builds(ResNet50_Weights.DEFAULT.transforms)

# Certain order of transforms matters 
cls_imb_s_transforms = builds(
    torchvision.transforms.Compose,
    [
        horizontal_flip,
        crops,
        to_tensor,
        cifar_10_norm,
    ]
)

cls_imb_l_transforms = builds(
    torchvision.transforms.Compose,
    [
        horizontal_flip,
        crops,
        to_tensor,
        cifar_100_norm,
    ]
)

sp_pix_transforms = builds(
    torchvision.transforms.Compose,
    [
        horizontal_flip,
        crops,
        to_tensor,
        mnist_norm
    ]
)

cifr10_datamdl_cfg = builds(CIFAR10DataModule)
cifr100_datamdl_cfg = builds(CIFAR100DataModule)
mnist_datamdl_cfg = builds(MNISTDataModule)

total_local_cores = 0
total_local_cores = total_local_cores if type(total_local_cores) == int else 1

cls_imb_datamdl_s_cfg = cifr10_datamdl_cfg(
    train_transforms = cls_imb_s_transforms,
    val_transforms = cls_imb_s_transforms,
    test_transforms = cls_imb_s_transforms,
    batch_size = 64,
    num_workers = total_local_cores
)

cls_imb_datamdl_l_cfg = cifr100_datamdl_cfg(
    train_transforms = cls_imb_l_transforms,
    val_transforms = cls_imb_l_transforms,
    test_transforms = cls_imb_l_transforms,
    batch_size = 64,
    num_workers = total_local_cores
)

sp_pix_datamdl_cfg = mnist_datamdl_cfg(
    train_transforms = sp_pix_transforms,
    val_transforms = sp_pix_transforms,
    test_transforms = sp_pix_transforms,
    batch_size = 64,
    num_workers = total_local_cores
)

sbg_rob_datamdl_cfg = mnist_datamdl_cfg(
    train_transforms = sp_pix_transforms,
    val_transforms = sp_pix_transforms,
    test_transforms = sp_pix_transforms,
    batch_size = 64,
    num_workers = total_local_cores
)

sp_pix_lbl = "sp_pix"
sbg_rob_lbl = "sbg_rob"
cls_imb_s_lbl = "cls_imb_s"
cls_imb_l_lbl = "cls_imb_l"
base_lbl = "base"
dummy_lbl = "dummy"
single_lbl = "single"
no_muon_lbl = "no_muon"

# Create groups: Key value pairs (At current granularity level desired to register options per group)
lit_module_store = store(group="lit_module")
lit_module_store(DualOptLitModule, name=f"{base_lbl}_dual")
lit_module_store(sp_pix_module_cfg, name=sp_pix_lbl)
lit_module_store(sbg_rob_module_cfg, name=sbg_rob_lbl)
lit_module_store(cls_imb_s_module_cfg, name=cls_imb_s_lbl)
lit_module_store(cls_imb_l_module_cfg, name=cls_imb_l_lbl)
# Tests single optimiser with specific params copied from dual optimizer

trainer_store = store(group="trainer")
trainer_store(trainer_cfg, name=base_lbl)

datamodule_store = store(group="datamodule")
datamodule_store(cls_imb_datamdl_s_cfg, name=cls_imb_s_lbl)
datamodule_store(cls_imb_datamdl_l_cfg, name=cls_imb_l_lbl)
datamodule_store(sp_pix_datamdl_cfg, name=sp_pix_lbl)
datamodule_store(sbg_rob_datamdl_cfg, name=sbg_rob_lbl)

model_store = store(group="model_cache")
model_store(cls_imb_model_s_cfg, name=cls_imb_s_lbl)
model_store(cls_imb_model_l_cfg, name=cls_imb_l_lbl)
model_store(sp_pix_model_cfg, name=sp_pix_lbl)
model_store(sbg_rob_model_cfg, name=sbg_rob_lbl)

params_store = store(group="params_cache")
params_store(cls_imb_s_params_cfg, name=cls_imb_s_lbl)
params_store(cls_imb_l_params_cfg, name=cls_imb_l_lbl)
params_store(sp_pix_params_cfg, name=sp_pix_lbl)
params_store(sbg_rob_params_cfg, name=sbg_rob_lbl)

schdlr_store = store(group="schdlrs_cache")
schdlr_store(lr_sch_pcfg, name=base_lbl)

grad_accum_schedule_store = store(group="grad_accum_schedule")
grad_accum_schedule_store(grad_accum_schedule_cfg, name=base_lbl)

optims_store = store(group="optims_cache")
optims_store(dual_optims_pcfg, name=cls_imb_s_lbl)
optims_store(dual_optims_pcfg, name=cls_imb_l_lbl)
optims_store(dual_optims_pcfg, name=sp_pix_lbl)
optims_store(dual_optims_pcfg, name=sbg_rob_lbl)
optims_store(single_optim_pcfg, name=single_lbl)

# Make a pure dict object like a class, but immediately configurable
ExperimentConfig = make_config(
    seed=1,
    hydra_defaults=[
        {'lit_module': sp_pix_lbl},
        {'params_cache': sp_pix_lbl},
        {'params_cache.use_muon': "False"},
        {'model_cache': sp_pix_lbl},
        {'datamodule': sp_pix_lbl},
        {'trainer': base_lbl},        
        {'optims_cache': sp_pix_lbl},
        {'schdlrs_cache': base_lbl},
        {'grad_accum_schedule': base_lbl},
        '_self_'
    ]
)

store(ExperimentConfig, name="adam_muon_benchmarks")
print("Example group entry", store["params_cache", sp_pix_lbl])
pre_seed = zen(lambda seed: pl.seed_everything(seed))
task_function = zen(train_and_val, pre_call=pre_seed)

if __name__ == "__main__":
    store.add_to_hydra_store()
    job = launch(
        ExperimentConfig,
        task_function,
        config_name="adam_muon_benchmarks",
        
        version_base="1.3",
        overrides=[
            f"model_cache={cls_imb_s_lbl}", 
            f"datamodule={cls_imb_s_lbl}", 
            f"params_cache={cls_imb_s_lbl}", 
            f"optims_cache={cls_imb_s_lbl}", 
            f"lit_module={cls_imb_s_lbl}",
            "params_cache.use_muon=False"
        ]
    )

