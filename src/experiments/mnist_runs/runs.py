from math import pi
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from hydra_zen import builds, make_config, make_custom_builds_fn, zen
import torch as tr
from muon import MuonWithAuxAdam
from torch.utils.data import DataLoader
from src.trainer.modules.class_imbalance_module import ClassImbLitModule 
from src.trainer.dataloader import mnist_dataloader
from src.trainer.per_optim_params.resnet_params.get_params import get_Muon_Aux_Adam_params
from src import train_and_val
from torchvision.models import resnet18, resnet50
from hydra_zen import instantiate, store
from hydra_zen.third_party.beartype import validates_with_beartype

# TODO: Import ExperimentConfig with same name, do not define here
# See file mnist_trainer.py where I have matching note and instructions, that is the task_eval function to adopt
# ExperimentConfig = make_config(
#     seed=1,
#     lit_module=lit_module,
#     trainer=class_imb_trainer,
# )


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