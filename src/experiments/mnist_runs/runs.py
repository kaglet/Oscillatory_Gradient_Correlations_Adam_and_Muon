from math import pi
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from hydra_zen import builds, make_config, make_custom_builds_fn, zen
import torch as tr
from muon import MuonWithAuxAdam
from torch.utils.data import DataLoader
from src.trainer.modules.class_imbalance_zen_module import ClassImbLitModule 
from src.trainer.dataloader import mnist_dataloader
from src.trainer.per_optim_params.resnet_params.get_params import get_Muon_Aux_Adam_params
from src import train_and_val
from torchvision.models import resnet18, resnet50
from hydra_zen import instantiate, store
from hydra_zen.third_party.beartype import validates_with_beartype

def get_callbacks() -> list[Callback]:
    """"
    Simplest workaround to define a list arg in config as Hydra is innately difficult with this. 
    
    TODO: 
    Later an extension could be to allow *args, and return those in a list format. 
    Input args to a function are configurable, then simply have function return the args as a list. 
    """
    model_checkpoint = ModelCheckpoint(
        save_top_k=3, 
        mode="max", 
        # TODO: Ensure unification across logging metric monitored. Unit 6.1 of PL docs do not cover ensuring consistency.
        # TODO: Change val accuracy to be the best on on a subgroup not the average

        monitor="val/acc", 
        save_last=True
    )

    return [model_checkpoint]

pbuilds = make_custom_builds_fn(
    zen_partial=True, 
    populate_full_signature=True,
    zen_wrappers=validates_with_beartype
)

model_store = store(group="model/")

# TODO: MUST adjust model based on 
class_imb_model_small = builds(resnet18) 
class_imb_model_big = builds(resnet50) 

model_store(class_imb_model_small, name="class_imb_model_small")
model_store(class_imb_model_big, name="class_imb_model_big")

class_imb_params = builds(
    get_Muon_Aux_Adam_params,
    model=class_imb_model_small,
    lr_each=(0.2, 0.01), 
    weight_decay_each=(0, 0), 
    betas=(0.95, 0.99), 
    use_layer_recs=False
)

muon_aux_adam_optim = pbuilds(MuonWithAuxAdam)

lit_module = builds(
    ClassImbLitModule, 
    model=class_imb_model_small,
    params=class_imb_params,
    optim=muon_aux_adam_optim,
)

optim_store = store(group="optim/")
optim_store(muon_aux_adam_optim, name="muon_aux_adam_optim")

params_store = store(group="params/")
params_store(class_imb_params, name="class_imb_params")

data_loader_store = store(group="dataloader/")
data_loader_store(class_imb_dataloader, name="class_imb_dataloader")


trainer_store = store(group="trainer/")

class_imb_trainer = builds(
    pl.Trainer, 
    max_epochs=100,
    deterministic=True,
    accelerator="auto",
    barebones=False,
    benchmark=True,
    callbacks=builds(get_callbacks),
    default_root_dir=os.getcwd(),
    detect_anomaly=False,
    enable_autolog_hparams=True,
)  

trainer_store(class_imb_trainer, name="class_imb_trainer")

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