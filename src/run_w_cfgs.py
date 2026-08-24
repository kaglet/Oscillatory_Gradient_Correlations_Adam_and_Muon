
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
from hydra_zen import instantiate, just, make_custom_builds_fn, make_config, make_custom_builds_fn, zen, ZenStore
from muon import MuonWithAuxAdam
from torch.optim import Muon, Adam
from torch.optim.lr_scheduler import LRScheduler
from src.train_and_val import train_and_val
from src.trainer.dataloader.cifar_dataloaders import CIFAR100DataModule, CIFAR10DataModule
from src.trainer.dataloader.components.mnist_normalization import mnist_normalization
from src.trainer.dataloader.mnist_dataloader import MNISTDataModule, SimpleMNISTDataModule
from src.trainer.modules.components.custom_grad_accum import GradAccumSchedule
from src.trainer.modules.general_lit_module import DualOptLitModule, SingleOptLitModule 
from src.trainer.modules.components.custom_resnet import create_resnet_model
from src.trainer.modules.components.simple_net import SimpleCNN
from src.trainer.per_optim_params.resnet_params.get_params import get_Adam_params, get_dist_Muon_Adam_params, get_nondist_Muon_Adam_params
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights
from hydra_zen.third_party.beartype import validates_with_beartype
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization

builds = make_custom_builds_fn(populate_full_signature=True)
pbuilds = make_custom_builds_fn(
    zen_partial=True, 
    populate_full_signature=True,
    zen_wrappers=validates_with_beartype
)

# TODO: For all below, set base cfg defaults or correct fill behaviour for error handling as this base cfg cannot be used as is
# ---- LIGHTNING MODULE CONFIGS BEGIN ---- #

# Define base params cfg
non_dist_params_cfg = pbuilds(
    get_nondist_Muon_Adam_params,
)

dist_params_cfg = pbuilds(
    get_dist_Muon_Adam_params,
)

# Define base vision model cfg
resnet_model_cfg = builds(create_resnet_model)
sp_pix_model_cfg = builds(SimpleCNN)

cls_imb_s_num_classes = 10
cls_imb_l_num_classes = 100
sbg_rob_num_classes = 5
sp_pix_num_classes = 10

# TODO: Add setting for num_in

# Class imbalance experiment: Define new instance overriding default values
cls_imb_model_s_cfg = resnet_model_cfg(net=just(resnet18), num_classes=cls_imb_s_num_classes, num_in=3)
cls_imb_model_l_cfg = resnet_model_cfg(net=just(resnet50), num_classes=cls_imb_l_num_classes) 
# Subgroup robustness experiment: Define new instance overriding default values
sbg_rob_model_cfg = resnet_model_cfg(net=just(resnet50), num_classes=sbg_rob_num_classes, num_in=1)

single_opt_resnet_model_s_cfg = resnet_model_cfg(net=just(resnet18), num_classes=10, num_in=1)

# TODO: Remove error for this possibly just Pylance not well configured for hydra-zen type checking
# Class imbalance experiment: Define new instance overriding default values
cls_imb_s_params_cfg = pbuilds(  
    get_nondist_Muon_Adam_params,
    lr_each=(1e-1, 1e-5), 
    weight_decay_each=(0, 0), 
    betas=(0.9, 0.99)
)

cls_imb_l_params_cfg = pbuilds(  
    get_nondist_Muon_Adam_params, 
    lr_each=(1e-1, 1e-5), 
    weight_decay_each=(0, 0), 
    betas=(0.9, 0.99)
)

sp_pix_params_cfg = pbuilds(   
    get_nondist_Muon_Adam_params,
    lr_each=(1e-1, 1e-3), 
    weight_decay_each=(0, 0), 
    betas=(0.9, 0.99)
)

single_opt_params_cfg = pbuilds(
    get_Adam_params,
    lr_each=(1e-1, 1e-5), 
    weight_decay_each=(0, 0), 
    betas=(0.9, 0.99)
)

# Subgroup robustness experiment: Define new instance overriding default values
sbg_rob_params_cfg = pbuilds(   
    get_nondist_Muon_Adam_params,
    lr_each=(1e-1, 1e-5), 
    weight_decay_each=(0, 0), 
    betas=(0.9, 0.99),
    use_layer_recs=False
)

# Define base optimiser cfg
mu_aux_adam_pcfg = [pbuilds(MuonWithAuxAdam)]
non_dist_optims_pcfg = [pbuilds(Muon), pbuilds(Adam)]
single_optim_pcfg = [pbuilds(Adam)]

dummy_schdlrs_pcfg = [pbuilds(LRScheduler), pbuilds(LRScheduler)]
# Class imbalance experiment: Define default (configurable) lit_module cfg  
cls_imb_s_module_cfg = pbuilds(
    DualOptLitModule, 
    grad_accum_schedule = builds(GradAccumSchedule, {0: 32}),
    num_classes = cls_imb_s_num_classes
)

cls_imb_l_module_cfg = pbuilds(
    DualOptLitModule, 
    grad_accum_schedule = builds(GradAccumSchedule, {0: 32}),
    num_classes = cls_imb_l_num_classes
)

sp_pix_module_cfg = pbuilds(
    DualOptLitModule, 
    grad_accum_schedule = builds(GradAccumSchedule, {0: 32}),
    num_classes = sp_pix_num_classes
)

# TODO: Fix this copied placeholder
sbg_rob_module_cfg = pbuilds(
    DualOptLitModule, 
    grad_accum_schedule = builds(GradAccumSchedule, {0: 32}),
    num_classes = sbg_rob_num_classes
)

test_single_opt_module_cfg = pbuilds(
    SingleOptLitModule, 
    grad_accum_schedule = builds(GradAccumSchedule, {0: 32}),
    num_classes = sp_pix_num_classes
)

# ---- LIGHTNING MODULE CONFIGS COMPLETE ---- #

# ---- TRAINER CONFIGS START ---- #

# Define base chkpt cfg
chkpt_cfg = builds(
    ModelCheckpoint,
) 

# TODO: Change val accuracy to be the best on on a subgroup not overall
# Class imbalance experiment: Define new instance overriding default values
cls_imb_chkpt_cfg = chkpt_cfg(    
    save_top_k=1, 
    mode="max", 
    # TODO: As Unit 6.1 of PL docs do not cover ensuring consistency, find out how across names.
    monitor="val/acc", 
    save_last=True
)

# Create static list of checkpoint behaviours, TODO: can be saved into group later
chkpts = [cls_imb_chkpt_cfg]

def create_callbacks(callbacks: list[Callback]) -> list[Callback]:
    """"
    Simplest workaround to define and configure lists as per Hydra-zen GitH discussion
    """
    return callbacks

# TODO: Fix this current error stopping file
# Define base chkpt cfg list returned, TODO: cfg can have non-generic instances to override later
create_cls_imb_clbks_cfg = builds(create_callbacks, callbacks=chkpts) 
create_sp_pix_clbks_cfg = create_cls_imb_clbks_cfg

# Define base trainer cfg
trainer_cfg = builds(
    pl.Trainer, 
    max_epochs=4,
    deterministic=False,
    accelerator="gpu",
    benchmark=False,
    callbacks=create_cls_imb_clbks_cfg,
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

# Create groups: Key value pairs (At current granularity level desired to register options per group)
store = ZenStore(deferred_hydra_store=True, overwrite_ok=True, deferred_to_config=True)
lit_module_store = store(group="experiment/lit_module/")
lit_module_store(DualOptLitModule, name=f"{base_lbl}_dual")
lit_module_store(SingleOptLitModule, name=f"{base_lbl}_single")
lit_module_store(sp_pix_module_cfg, name=sp_pix_lbl)
lit_module_store(sbg_rob_module_cfg, name=sbg_rob_lbl)
lit_module_store(cls_imb_s_module_cfg, name=cls_imb_s_lbl)
lit_module_store(cls_imb_l_module_cfg, name=cls_imb_l_lbl)
# Tests single optimiser with specific params copied from dual optimizer
lit_module_store(test_single_opt_module_cfg, name="dummy_single")

trainer_store = store(group="experiment/trainer/")
trainer_store(trainer_cfg, name=f"{base_lbl}")

datamodule_store = store(group="experiment/datamodule/")
datamodule_store(cls_imb_datamdl_s_cfg, name=cls_imb_s_lbl)
datamodule_store(cls_imb_datamdl_l_cfg, name=cls_imb_l_lbl)
datamodule_store(sp_pix_datamdl_cfg, name=sp_pix_lbl)
datamodule_store(sbg_rob_datamdl_cfg, name=sbg_rob_lbl)

model_store = store(group="experiment/model_cache/")
model_store(cls_imb_model_s_cfg, name=cls_imb_s_lbl)
model_store(cls_imb_model_l_cfg, name=cls_imb_l_lbl)
model_store(sp_pix_model_cfg, name=sp_pix_lbl)
model_store(sbg_rob_model_cfg, name=sbg_rob_lbl)

params_store = store(group="experiment/params_cache/")
params_store(cls_imb_s_params_cfg, name=cls_imb_s_lbl)
params_store(cls_imb_l_params_cfg, name=cls_imb_l_lbl)
params_store(sp_pix_params_cfg, name=sp_pix_lbl)
params_store(sbg_rob_params_cfg, name=sbg_rob_lbl)

schdlr_store = store(group="experiment/schdlrs_cache/")

grad_accum_schedule = store(group="experiment/grad_acc_schdlrs/")

optims_store = store(group="experiment/optims_cache/")
optims_store(non_dist_optims_pcfg, name=cls_imb_s_lbl)
optims_store(non_dist_optims_pcfg, name=cls_imb_l_lbl)
optims_store(non_dist_optims_pcfg, name=sp_pix_lbl)
optims_store(non_dist_optims_pcfg, name=sbg_rob_lbl)

# Make a pure dict object like a class, but immediately configurable
ExperimentConfig = make_config(
    seed=1,
    lit_module=sp_pix_module_cfg,
    trainer=trainer_cfg,
    datamodule=sp_pix_datamdl_cfg,
    model_cache=sp_pix_model_cfg,
    params_cache=sp_pix_params_cfg,
    optims_cache = non_dist_optims_pcfg,
    schdlrs_cache = dummy_schdlrs_pcfg
)

store(ExperimentConfig, name="adam_muon_benchmarks", group="experiment")
store.add_to_hydra_store()
print("Example group entry", store["experiment/params_cache/", sp_pix_lbl])
pre_seed = zen(lambda seed: pl.seed_everything(seed))
task_function = zen(train_and_val, pre_call=pre_seed)

# Allows CLI run
if __name__ == "__main__":
    task_function.hydra_main(
        config_name="adam_muon_benchmarks",
        version_base="1.2",
        config_path=".",
    )
