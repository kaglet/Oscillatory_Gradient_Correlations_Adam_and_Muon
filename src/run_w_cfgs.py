
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
from trainer.modules.general_lit_module import DualOptLitModule, SingleOptLitModule 
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
simple_cnn_cfg = builds(SimpleCNN)

cls_imb_s_num_classes = 10
cls_imb_l_num_classes = 100
sbg_rob_num_classes = 5
sp_pix_num_classes = 10

cls_imb_model_s_cfg = resnet18_model_cfg(num_classes=cls_imb_s_num_classes, num_in=3)
cls_imb_model_l_cfg = resnet50_model_cfg(num_classes=cls_imb_l_num_classes, num_in=3) 
sbg_rob_model_cfg = resnet50_model_cfg(num_classes=sbg_rob_num_classes, num_in=3)
sp_pix_model_cfg = simple_cnn_cfg(num_classes=sp_pix_num_classes, num_in=1)

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

dual_lit_module_cfg = pbuilds(DualOptLitModule)
single_lit_module_cfg = pbuilds(SingleOptLitModule)

# Lighning Module variations (for groups mainly for swap in and out, TODO: Defer, ideas below)
# Top level DualOptLit module cannot be changed, the top can be registered or even try dictionary keys to unpack as keyword arguments
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

# ---- DATAMODULE CONFIGS START ---- #
#
horizontal_flip_cfg = builds(torchvision.transforms.RandomHorizontalFlip)
vertical_flip_cfg = builds(torchvision.transforms.RandomVerticalFlip)
crops_cfg = builds(torchvision.transforms.RandomCrop)
to_tensor_cfg = builds(torchvision.transforms.ToTensor)

cifar_10_norm_cfg = builds(cifar10_normalization)
cifar_100_norm_cfg = builds(imagenet_normalization)
mnist_norm_cfg = builds(mnist_normalization)

# Certain order of transforms matters 
cls_imb_s_transforms = builds(
    torchvision.transforms.Compose,
    [
        horizontal_flip_cfg,
        crops_cfg(size=20),
        to_tensor_cfg,
        cifar_10_norm_cfg,
    ]
)

cls_imb_l_transforms = builds(
    torchvision.transforms.Compose,
    [
        horizontal_flip_cfg,
        crops_cfg(size=20),
        to_tensor_cfg,
        cifar_100_norm_cfg,
    ]
)

sp_pix_transforms = builds(
    torchvision.transforms.Compose,
    [
        horizontal_flip_cfg,
        crops_cfg(size=20),
        to_tensor_cfg,
        mnist_norm_cfg
    ]
)

sbg_rob_transforms = builds(
    torchvision.transforms.Compose,
    [
        horizontal_flip_cfg,
        crops_cfg(size=20),
        to_tensor_cfg,
        mnist_norm_cfg
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

# TODO: Store these overlapping keys in a dictionary rather to not keep repeating prefix values and to make config easier
default_lbl = "default"
single_lbl = "single"
dual_lbl = "dual"

# Create groups: Key value pairs (At current granularity level desired to register options per group)
lit_module_store = store(group="lit_module")
lit_module_store(dual_lit_module_cfg, name=dual_lbl)
lit_module_store(single_lit_module_cfg, name=single_lbl)

trainer_store = store(group="trainer")
trainer_store(trainer_cfg, name=default_lbl)

sp_pix_lbl = "sp_pix"
sbg_rob_lbl = "sbg_rob"
cls_imb_s_lbl = "cls_imb_s"
cls_imb_l_lbl = "cls_imb_l"
datamodule_store = store(group="datamodule")
datamodule_store(cls_imb_datamdl_s_cfg, name=cls_imb_s_lbl)
datamodule_store(cls_imb_datamdl_l_cfg, name=cls_imb_l_lbl)
datamodule_store(sp_pix_datamdl_cfg, name=sp_pix_lbl)
datamodule_store(sbg_rob_datamdl_cfg, name=sbg_rob_lbl)

model_store = store(group="model_cache")
model_store(resnet18_model_cfg, name="resnet18")
model_store(resnet50_model_cfg, name="resnet50")
model_store(simple_cnn_cfg, name="simple_cnn")

params_store = store(group="params_cache")
params_store(dual_optim_params_cfg, name=dual_lbl)
params_store(single_optim_params_cfg, name=single_lbl)

schdlr_store = store(group="schdlrs_cache")
schdlr_store(lr_sch_pcfg, name=default_lbl)

grad_accum_schedule_store = store(group="grad_accum_schedule")
grad_accum_schedule_store(grad_accum_schedule_cfg, name=default_lbl)

optims_store = store(group="optims_cache")
optims_store(dual_optims_pcfg, name=dual_lbl)
optims_store(single_optim_pcfg, name=single_lbl)

# Make a pure dict object like a class, but immediately configurable
ExperimentConfig = make_config(
    seed=1,
    hydra_defaults=[
        {'lit_module': dual_lbl},
        {'params_cache': dual_lbl},
        {'model_cache': "resnet18"},
        {'datamodule': sp_pix_lbl},
        {'trainer': default_lbl},        
        {'optims_cache': dual_lbl},
        {'schdlrs_cache': default_lbl},
        {'grad_accum_schedule': default_lbl},
        '_self_'
    ]
)

store(ExperimentConfig, name="adam_muon_benchmarks")
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
            f"model_cache=resnet18", 
            f"datamodule={cls_imb_s_lbl}", 
            f"params_cache={dual_lbl}", 
            f"optims_cache={dual_lbl}", 
            f"lit_module={dual_lbl}",
            "++params_cache.lr_muon=0.02",
            "++params_cache.lr_adam=0.001",
            "++params_cache.momentum_muon=0.2",
            "++params_cache.beta_1_adam=0.95",
            "++params_cache.beta_2_adam=0.99",
            "++params_cache.nesterov=True",
            "++params_cache.use_layer_recs=False",
            "++params_cache.eps=1e-10",
        ]
    )

