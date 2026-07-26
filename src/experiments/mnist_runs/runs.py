from math import pi
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from hydra_zen import builds, make_config, make_custom_builds_fn, zen
import torch as tr
from muon import MuonWithAuxAdam
from torch.utils.data import DataLoader
from src.trainer.modules.class_imbalance_zen_module import ClassImbLitModule 
from src.trainer.dataloader import mnist_zen_dataloader
from src.trainer.per_optim_params.resnet_params.get_params import get_Muon_Aux_Adam_params
from src import train_and_val
from torchvision.models import resnet18, resnet50
from hydra_zen import instantiate, store
from hydra_zen.third_party.beartype import validates_with_beartype

# Groups (Uncertainty for level, but groups are simply for nice access): OptimConf, DataloaderConf, TrainerConf, LitModule
# Anything you want configurable in parameters use builds I.e. No use object literals, rather get the config value. 
pbuilds = make_custom_builds_fn(
    zen_partial=True, 
    populate_full_signature=True,
    zen_wrappers=validates_with_beartype
)

def get_callbacks() -> list[Callback]:
    """"
    Simplest workaround as Hydra is difficult with configuring lists. 
    
    - Later an extension could be to allow *args, and return those in a list format. 
    Input args to a function are configurable, then simply have function return the args as a list. 
    """
    model_checkpoint = ModelCheckpoint(
        save_top_k=3, 
        mode="max", 
        # TODO: Ensure unification across logging metric monitored. Unit 6.1 of PL docs do not cover ensuring consistency.
        monitor="val/acc", 
        save_last=True
    )

    return [model_checkpoint]

callbacks = get_callbacks

class_imb_trainer = builds(
    pl.Trainer, 
    max_epochs=100,
    deterministic=True,
    accelerator="auto",
    devices=1,
    # Turn off to remove useful overhead in training and debugging 
    barebones=False,
    # Fixed input size (image pixel dims) therefore below is for speedup
    benchmark=True,
    # TODO: Add callbacks
    # TODO: Change val accuracy to be the best on on a subgroup not the average
    # TODO: Finish the val accuracy log in Lightning Module
    callbacks=builds(get_callbacks),
    # TODO: Change this default for cluster training, use env variable
    default_root_dir=os.getcwd(),
    # Disable below only for debugging, massive slowdown result.
    detect_anomaly=False,
    # Set to false and follow docs to set custom hparams (hyperparameters) log
    enable_autolog_hparams=True,
    # TODO: Continue skim remaining Trainer arg defaults and expose them for config or simple train with defaults and look for guides for important stuff
)  

trainer_store = store(group="trainer/")
trainer_store(class_imb_trainer, name="class_imb_trainer")

class_imb_model_small = builds(resnet18) 
class_imb_model_big = builds(resnet50) 

model_store = store(group="model/")
model_store(class_imb_model_small, name="class_imb_model_small")
model_store(class_imb_model_big, name="class_imb_model_big")

# TODO: Make static groups of parameters mapping to the figure in the paper
# TODO: Model missing, access from config store later, since depends on config built
# Make subconfigs of class imbalance params, numbered from 1 to n
class_imb_params = builds(
    get_Muon_Aux_Adam_params,
    # Lightning module returns fields we desire, but this would be circular.
    # So reuse the same model instance built, even if reference not shared.
    model=class_imb_model_small,
    lr_each=(0.2, 0.01), 
    weight_decay_each=(0, 0), 
    betas=(0.95, 0.99), 
    use_layer_recs=False
)

params_store = store(group="params/")
params_store(class_imb_params, name="class_imb_params")

class_imb_dataloader = pbuilds(
    DataLoader, 
    batch_size=2048, 
    shuffle=True, 
    drop_last=False
)

params_store = store(group="params/")
params_store(class_imb_params, name="class_imb_params")

optim = pbuilds(MuonWithAuxAdam)
# Not necessary as we always use MuonWithAuxAdam, but for later swops if curious
params_store = store(group="params/")
params_store(class_imb_params, name="class_imb_params")

"""
TODO: Finish defining defaults THEN define groups. 
"""
lit_module = pbuilds(
    ClassImbLitModule, 
    model=class_imb_model_small,
    params=class_imb_params,
    optim=optim,
    dataloader=class_imb_dataloader
)
params_store = store(group="lit_module/")
params_store(class_imb_params, name="class_imb_params")


# TODO: Build this based on passed in/built architecture to the Trainer I believe


# TODO: FIX passing everything into here instead of already built trainer and lit_module
# See file mnist_trainer.py where I have matching note and instructions, that is the task_eval function to adopt
ExperimentConfig = make_config(
    seed=1,
    lit_module=lit_module,
    trainer=class_imb_trainer,
)

# Wrapping `train_and_eval` with `zen` makes it compatible with Hydra as a task function
#
# We must specify `pre_call` to ensure that pytorch lightning seeds everything
# *before* any of our configs are instantiated (which will initialize the pytorch
# model whose weights depend on the seed)
pre_seed = zen(lambda seed: pl.seed_everything(seed, workers=True))
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