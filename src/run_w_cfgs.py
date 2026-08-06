"""
Anything configurable or static in a configurable aspect is defined here and assembled later.
"""

import os
from typing import Any, Type

import torchvision
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from hydra_zen import builds, make_config, make_custom_builds_fn, zen
from muon import MuonWithAuxAdam
from src.train_and_val import train_and_val
from src.trainer.dataloader.cifar_dataloaders import CIFAR100DataModule, CIFAR10DataModule
from src.trainer.dataloader.components.mnist_normalization import mnist_normalization
from src.trainer.dataloader.mnist_dataloader import MNISTDataModule
from src.trainer.modules.general_lit_module import GeneralLitModule 
from src.trainer.modules.components.custom_resnet import create_resnet_model
from src.trainer.modules.components.simple_net import SimpleCNN
from src.trainer.per_optim_params.resnet_params.get_params import get_Muon_Aux_Adam_params
from torchvision.models import resnet18, resnet50
from hydra_zen import store
from hydra_zen.third_party.beartype import validates_with_beartype
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization


pbuilds = make_custom_builds_fn(
    zen_partial=True, 
    populate_full_signature=True,
    zen_wrappers=validates_with_beartype
)

# TODO: For all below, set base cfg defaults or correct fill behaviour for error handling as this base cfg cannot be used as is
# ---- LIGHTNING MODULE CONFIGS BEGIN ---- #

# Define base params cfg
params_cfg = builds(
    get_Muon_Aux_Adam_params,
)

# Define base vision model cfg
resnet_model_cfg = builds(create_resnet_model)
sp_pix_model_cfg = builds(SimpleCNN)

# Class imbalance experiment: Define new instance overriding default values
cls_imb_model_s_cfg = resnet_model_cfg(resnet18)
cls_imb_model_l_cfg = resnet_model_cfg(resnet50, num_classes=100) 

# Subgroup robustness experiment: Define new instance overriding default values
sbg_rob_model_clb_cfg = resnet_model_cfg(resnet50)

# Class imbalance experiment: Define new instance overriding default values
cls_imb_params_cfg = params_cfg(   
    model=resnet_model_cfg,
    lr_each=(0.2, 0.01), 
    weight_decay_each=(0, 0), 
    betas=(0.95, 0.99), 
    use_layer_recs=False
)

# Subgroup robustness experiment: Define new instance overriding default values
sbg_rob_params_cfg = params_cfg(   
    model=resnet_model_cfg,
    lr_each=(0.2, 0.01), 
    weight_decay_each=(0, 0), 
    betas=(0.95, 0.99), 
    use_layer_recs=False
)

# Define base optimiser cfg
mu_aux_adam_pcfg = pbuilds(MuonWithAuxAdam)

# Class imbalance experiment: Define default (configurable) lit_module cfg  
cls_imb_module_cfg = builds(
    GeneralLitModule, 
    model=cls_imb_model_s_cfg,
    params=cls_imb_params_cfg,
    optim=mu_aux_adam_pcfg,
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

def create_clbks(*callbacks: Callback) -> list[Callback]:
    """"
    Simplest workaround to define and configure lists as per Hydra-zen GitH discussion
    """
    return list(callbacks)

# Define base chkpt cfg list returned, TODO: cfg can have non-generic instances to override later
create_clbks_cfg = builds(create_clbks, callbacks=chkpts) 

# Define base trainer cfg
trainer_cfg = builds(
    pl.Trainer, 
    max_epochs=200,
    deterministic=True,
    accelerator="auto",
    barebones=False,
    benchmark=True,
    callbacks=create_clbks_cfg,
    default_root_dir=os.getcwd(),
    detect_anomaly=False,
    enable_autolog_hparams=True,
)  

# ---- TRAINER CONFIGS COMPLETE ---- #
# ---- DATAMODULE CONFIGS START ---- #
cifar10_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar100_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # TODO: Find out if this stand-in is fine, from uh reddit commenter, or if to calc manually 
        imagenet_normalization(),
    ]
)

cifr10_dmodule_cfg = builds(CIFAR10DataModule)
cifr100_dmodule_cfg = builds(CIFAR100DataModule)
mnist_dmodule_cfg = builds(MNISTDataModule)

cls_imb_dmodule_s_cfg = cifr10_dmodule_cfg(
    train_transforms = cifar10_transforms,
    val_transforms = cifar10_transforms,
    test_transforms = cifar10_transforms,
    batch_size = 2048,
    num_workers = 4
)

cls_imb_dmodule_l_cfg = cifr100_dmodule_cfg(
    train_transforms = cifar100_transforms,
    val_transforms = cifar100_transforms,
    test_transforms = cifar100_transforms,
    batch_size = 2048,
    num_workers = 4
)

mnist_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(), 
        mnist_normalization()
    ]
)

sp_pix_dmodule_cfg = mnist_dmodule_cfg(
    train_transforms = mnist_transforms,
    val_transforms = mnist_transforms,
    test_transforms = mnist_transforms,
    batch_size = 2048,
    num_workers = 4
)

# TODO: Import only this
# TODO: See file mnist_trainer.py where I have matching note and instructions, that is the task_eval function to adopt
# Make a pure dict object like a class, but immediately configurable
ExperimentConfig = make_config(
    seed=1,
    lit_module=cls_imb_module_cfg,
    trainer=trainer_cfg,
    dmodule=cls_imb_model_s_cfg
)

pre_seed = zen(lambda seed: pl.seed_everything(seed))
task_function = zen(train_and_val, pre_call=pre_seed)

if __name__ == "__main__":
    # enables us to call
    from hydra_zen import ZenStore

    store = ZenStore(deferred_hydra_store=False)
    store(ExperimentConfig, name="lit_app")

    task_function.hydra_main(
        config_name="lit_app",
        version_base="1.1",
        config_path=".",
    )
