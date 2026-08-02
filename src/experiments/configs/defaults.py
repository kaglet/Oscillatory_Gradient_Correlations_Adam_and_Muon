import os

import torchvision
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from hydra_zen import builds, make_config, make_custom_builds_fn, zen
from muon import MuonWithAuxAdam
from src.trainer.modules.class_imbalance_zen_module import ClassImbLitModule 
from src.trainer.per_optim_params.resnet_params.get_params import get_Muon_Aux_Adam_params
from torchvision.models import resnet18, resnet50
from hydra_zen import store
from hydra_zen.third_party.beartype import validates_with_beartype
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

"""
To more easily keep track:
We define defaults at a particular parameter nesting level.
Then we define the group nesting level that unifies them during composition. 

Intermediately:
Groups are then outsourced to external functions in groups.py 
with a unified parent group name arg, and the param name 
auto-derived to remove human error. 
"""

def get_callbacks(callback: Callback, *callbacks: Callback) -> list[Callback]:
    """"
    Simplest workaround to define a list arg in config as Hydra is innately difficult with this. 
    TODO: 
    Make arbitrary number of input args to a function that are configurable, then simply have function return the args as a list. 
    """
    return [callback]

def get_transforms(transform: nn.Module, *transforms: nn.Module) -> list[nn.Module]:
    return [transform]

pbuilds = make_custom_builds_fn(
    zen_partial=True, 
    populate_full_signature=True,
    zen_wrappers=validates_with_beartype
)

"""
TODO: 
- Allow nesting on class imb by having it as a dictionary for example, via configs. 
- Ensure object literals are renamed to cfg when that is required such as in groups etc., including transforms.
"""
def get_lit_module():
    class_imb_params = builds(
        get_Muon_Aux_Adam_params,
        # TODO: Fix and finalize the below interpolation
        model="${lit_module.model.class_imb_model_small}",
        lr_each=(0.2, 0.01), 
        weight_decay_each=(0, 0), 
        betas=(0.95, 0.99), 
        use_layer_recs=False
    )

    muon_aux_adam_optim = pbuilds(MuonWithAuxAdam)

    lit_module = builds(
        ClassImbLitModule, 
        model="${lit_module.model.class_imb_model_small}",
        params=class_imb_params,
        optim=muon_aux_adam_optim,
    )

    return lit_module 

def get_trainer():
    model_checkpoint = builds(
        ModelCheckpoint,
        save_top_k=3, 
        mode="max", 
        # TODO: Ensure unification across logging metric monitored. Unit 6.1 of PL docs do not cover ensuring consistency.
        # TODO: Change val accuracy to be the best on on a subgroup not the average
        monitor="val/acc", 
        save_last=True
    ) 

    get_callbacks_cfg = builds(get_callbacks, callbacks=model_checkpoint) 

    class_imb_trainer = builds(
        pl.Trainer, 
        max_epochs=100,
        deterministic=True,
        accelerator="auto",
        barebones=False,
        benchmark=True,
        callbacks=get_callbacks_cfg,
        default_root_dir=os.getcwd(),
        detect_anomaly=False,
        enable_autolog_hparams=True,
    )  

    return class_imb_trainer

def get_dataloader(): 
    # TODO: I don't want to build the below, as it gets too granular, so perhaps store in config store
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
class_imb_model_big = builds(resnet50) 
model_store(class_imb_model_small, name="class_imb_model_small")
model_store(class_imb_model_big, name="class_imb_model_big")

optim_store = store(group="experiment/lit_module/optim/")
optim_store(muon_aux_adam_optim, name="muon_aux_adam_optim")

params_store = store(group="experiment/lit_module/params/")
params_store(class_imb_params, name="class_imb_params")

class_imb_transforms_store(class_imb_train_transforms, name="class_imb_train_transforms")
class_imb_transforms_store(class_imb_val_transforms, name="class_imb_val_transforms")
class_imb_transforms_store(class_imb_test_transforms, name="class_imb_test_transforms")
get_transforms_cfg = builds(get_transforms)

# TODO: Import only this
# TODO: See file mnist_trainer.py where I have matching note and instructions, that is the task_eval function to adopt
# Make a pure dict object like a class, but immediately configurable
ExperimentConfig = make_config(
    seed=1,
    lit_module=lit_module,
    trainer=class_imb_trainer,
    transforms=,
)


