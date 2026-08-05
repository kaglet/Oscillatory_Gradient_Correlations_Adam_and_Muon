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
from src.trainer.modules.class_imbalance_module import ClassImbLitModule 
from src.trainer.per_optim_params.resnet_params.get_params import get_Muon_Aux_Adam_params
from torchvision.models import resnet18, resnet50
from hydra_zen import store
from hydra_zen.third_party.beartype import validates_with_beartype
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

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

# Class imbalance experiment: Define new instance overriding default values
cls_imb_params_cfg = params_cfg(   
    # TODO: Fix and finalize the below interpolation
    model="${lit_module.model.class_imb_model_small}",
    lr_each=(0.2, 0.01), 
    weight_decay_each=(0, 0), 
    betas=(0.95, 0.99), 
    use_layer_recs=False
)

# Subgroup robustness experiment: Define new instance overriding default values
sbg_rob_params_cfg = params_cfg(   
    # Use the model chosen for the task at group config time, where it is sourced from
    model="${lit_module.model.class_imb_model_small}",
    lr_each=(0.2, 0.01), 
    weight_decay_each=(0, 0), 
    betas=(0.95, 0.99), 
    use_layer_recs=False
)

# Define base optimiser cfg
# TODO: Extendable by defining a create function or class to define cfg overrides e.g. for groups
mu_aux_adam_pcfg = pbuilds(MuonWithAuxAdam)

# TODO: Move this up to model definition prior
# TODO: Edit type checked of resnet in fact a callable/mapping
def create_resnet_model(net: Any = resnet18, num_classes: int = 10, num_in: int=3, num_out: int = 64, kern_size: tuple[int, int] = (3,3), stride: tuple[int, int] = (1,1), padding: tuple[int, int] = (1,1), bias: bool = False) -> nn.Module:
    """ 
    Architecture dependent function to create and minimally edit a given resnet model. 
    """
    model = net(pretrained=False, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=kern_size, stride=stride, padding=padding, bias=bias)
    model.maxpool = nn.Identity()
    return model

def create_sp_pix_model(num_in: int = 1)  -> nn.Module:
    # TODO: Alter this based off definitions of other vision models, and of architecture used in paper 
    model = nn.Sequential(
        nn.Linear(1, num_neurons),
        nn.Sigmoid(),
        nn.Linear(num_neurons, 1, bias=False),
    )
    return model

# Define base vision model cfg
resnet_model_cfg = builds(create_resnet_model)
sp_pix_model_cfg = builds(create_sp_pix_model)

# Class imbalance experiment: Define new instance overriding default values
cls_imb_model_s_cfg = resnet_model_cfg(resnet18)
cls_imb_model_l_cfg = resnet_model_cfg(resnet50, num_classes=100) 

# Subgroup robustness experiment: Define new instance overriding default values
sbg_rob_model_clb_cfg = resnet_model_cfg(resnet50)

# Class imbalance experiment: Define default (configurable) lit_module cfg  
cls_imb_module_cfg = builds(
    ClassImbLitModule, 
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
    max_epochs=100,
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
class_imb_train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

class_imb_val_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

class_imb_test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

def mnist_normalization():
    return torchvision.transforms.Normalize(
        (0.1307,), (0.3081,)
    )

sp_pix_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(), 
        mnist_normalization()
    ]
)

def create_tfrms(*transforms: nn.Module) -> list[nn.Module]:
    return list(transforms)

class_imb_transforms_store = store(group="experiment/transforms")


lit_module_store = store(group="experiment/lit_module/")
lit_module_store(lit_module, name="class_imb_module")

trainer_store = store(group="experiment/trainer/")
trainer_store(class_imb_trainer, name="class_imb_trainer")

experiment_store = store(group="experiments")
subgrp_robust_store = experiment_store()
class_imb_store = experiment_store()
spurious_pix_store = experiment_store()

model_store = store(group="experiment/lit_module/model/")
model_store(class_imb_model_small, name="class_imb_model_small")
model_store(class_imb_model_big, name="class_imb_model_big")

optim_store = store(group="experiment/lit_module/optim/")
optim_store(muon_aux_adam_optim, name="muon_aux_adam_optim")

params_store = store(group="experiment/lit_module/params/")
params_store(class_imb_params, name="class_imb_params")

class_imb_transforms_store(class_imb_train_transforms, name="class_imb_train_transforms")
class_imb_transforms_store(class_imb_val_transforms, name="class_imb_val_transforms")
class_imb_transforms_store(class_imb_test_transforms, name="class_imb_test_transforms")
get_transforms_cfg = builds(create_tfrms)

# TODO: Import only this
# TODO: See file mnist_trainer.py where I have matching note and instructions, that is the task_eval function to adopt
# Make a pure dict object like a class, but immediately configurable
ExperimentConfig = make_config(
    seed=1,
    lit_module=lit_module,
    trainer=class_imb_trainer,
    transforms=,
)


