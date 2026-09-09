
"""
Anything configurable or static in a configurable aspect is defined here and assembled later.
"""

from dataclasses import dataclass
import os
import multiprocessing
from pathlib import Path
from typing import Any, Type

from omegaconf import OmegaConf
import torch
import torchvision
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from hydra_zen import instantiate, just, launch, make_custom_builds_fn, make_config, make_custom_builds_fn, multirun, zen, store
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.optim import Muon, Adam
from torch.optim.lr_scheduler import LRScheduler

from src.class_imbalance_ratios.datamodule.cls_imb_datamodule import ClsImbDataModule
from src.base.datamodule.components.mnist_normalization import mnist_normalization
from src.base.datamodule.mnist_datamodule import MNISTDataModule, SimpleMNISTDataModule
from src.base.lit_module.components.custom_grad_accum import GradAccumSchedule
from src.base.lit_module.components.custom_resnet import create_resnet_model
from src.base.lit_module.components.per_optim_params.resnet_params.get_params import get_Adam_params, get_dist_Muon_Adam_params, get_dual_opt_params, get_nondist_Muon_Adam_params
from src.base.lit_module.components.simple_net import SimpleCNN
from src.base.lit_module.general_lit_module import DualOptLitModule, SingleOptLitModule
from src.train_and_val import train_and_val
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
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

adam_optim_params_cfg = pbuilds(
    get_Adam_params,
)

single_optim_params_cfg = pbuilds(
    get_nondist_Muon_Adam_params,
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
dist_mu_aux_adam_pcfg = [pbuilds(MuonWithAuxAdam)]
# Custom for non-distributed
dual_optims_pcfg = [pbuilds(Muon), pbuilds(Adam)]
# Single optimiser training  
single_adam_pcfg = [pbuilds(Adam)]
# Single device 
single_mu_aux_adam_pcfg = [pbuilds(SingleDeviceMuonWithAuxAdam)]

# Gradient accum schedules 
grad_accum_schedule_cfg = builds(GradAccumSchedule, {0: 16})

# LR Schedulers
lr_sch_pcfg = [pbuilds(LRScheduler), pbuilds(LRScheduler)]

dual_lit_module_cfg = pbuilds(DualOptLitModule)
single_lit_module_cfg = pbuilds(SingleOptLitModule)

# Lighning Module variations (for groups mainly for swap in and out, TODO: Defer, ideas below)
# Top level DualOptLit module cannot be changed, the top can be registered or even try dictionary keys to unpack as keyword arguments
cls_imb_s_module_cfg = single_lit_module_cfg(
    grad_accum_schedule = grad_accum_schedule_cfg,
    num_classes = cls_imb_s_num_classes
)

cls_imb_l_module_cfg = pbuilds(
    SingleOptLitModule, 
    grad_accum_schedule = grad_accum_schedule_cfg,
    num_classes = cls_imb_l_num_classes
)

sp_pix_module_cfg = pbuilds(
    SingleOptLitModule, 
    grad_accum_schedule = grad_accum_schedule_cfg,
    num_classes = sp_pix_num_classes
)

sbg_rob_module_cfg = pbuilds(
    SingleOptLitModule, 
    grad_accum_schedule = grad_accum_schedule_cfg,
    num_classes = sbg_rob_num_classes
)

# ---- TRAINER CONFIGS START ---- #
# TODO: Change val accuracy to be the best on on a subgroup not overall
max_checkpt_cfg = pbuilds(
    ModelCheckpoint,    
    save_top_k=1, 
    mode="max", 
    # TODO: As Unit 6.1 of PL docs do not cover ensuring consistency, find out how across names.
    monitor="val/acc", 
    save_last=True,
)

chkpts_cfg_list = [max_checkpt_cfg]

tb_logger_cfg = pbuilds(TensorBoardLogger, name="per_epoch_metrics")
csv_logger_cfg = [
    pbuilds(CSVLogger, name="train_steps"),
    pbuilds(CSVLogger, name="val_steps"), 
    pbuilds(CSVLogger, name="val_max"), 

    pbuilds(CSVLogger, name="train_loss_steps"),
    pbuilds(CSVLogger, name="val_loss_steps"),

    pbuilds(CSVLogger, name="train_acc_epoch"),
    pbuilds(CSVLogger, name="train_loss_epoch"),
    pbuilds(CSVLogger, name="val_acc_epoch"), 
    pbuilds(CSVLogger, name="val_loss_epoch"),
]

# Define base trainer cfg
trainer_cfg = pbuilds(
    pl.Trainer, 
    max_epochs=10,
    deterministic=False,
    accelerator="gpu",
    benchmark=False,
    detect_anomaly=False,
    devices=1, 
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
        crops_cfg(size=28),
        to_tensor_cfg,
    ]
)

cls_imb_l_transforms = builds(
    torchvision.transforms.Compose,
    [
        horizontal_flip_cfg,
        crops_cfg(size=20),
        to_tensor_cfg,
    ]
)

sp_pix_transforms = builds(
    torchvision.transforms.Compose,
    [
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

cifr10_datamdl_cfg = builds(
    ClsImbDataModule, 
    downsampler_ratio=20, 
    weighted_ratios=None,     
)

cifr100_datamdl_cfg = builds(
    ClsImbDataModule, 
    downsampler_ratio=20, 
    weighted_ratios=None
)
mnist_datamdl_cfg = builds(MNISTDataModule)

total_local_cores = 0
total_local_cores = total_local_cores if type(total_local_cores) == int else 1

cls_imb_datamdl_s_cfg = cifr10_datamdl_cfg(
    train_transforms = cls_imb_s_transforms,
    val_transforms = cls_imb_s_transforms,
    test_transforms = cls_imb_s_transforms,
    batch_size = 128, # From batch size set grad_accum to get new effective batch size
    num_workers = 8
)

cls_imb_datamdl_l_cfg = cifr100_datamdl_cfg(
    train_transforms = cls_imb_l_transforms,
    val_transforms = cls_imb_l_transforms,
    test_transforms = cls_imb_l_transforms,
    batch_size = 128,
    num_workers = 8
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
mu_aux_ad_lbl = "mu_aux_ad"
dual_lbl = "dual"

# Create groups: Key value pairs (At current granularity level desired to register options per group)
lit_module_store = store(group="lit_module")
lit_module_store(dual_lit_module_cfg, name=dual_lbl)
lit_module_store(single_lit_module_cfg, name=mu_aux_ad_lbl)

trainer_store = store(group="trainer")
trainer_store(trainer_cfg, name=default_lbl)

sp_pix_lbl = "sp_pix"
sbg_rob_lbl = "sbg_rob"
cls_imb_s_lbl = "cls_imb_s"
cls_imb_l_lbl = "cls_imb_l"

lit_module_store(cls_imb_s_module_cfg, name=cls_imb_s_lbl)

datamodule_store = store(group="datamodule")
datamodule_store(cls_imb_datamdl_s_cfg, name=cls_imb_s_lbl)
datamodule_store(cls_imb_datamdl_l_cfg, name=cls_imb_l_lbl)
datamodule_store(sp_pix_datamdl_cfg, name=sp_pix_lbl)
datamodule_store(sbg_rob_datamdl_cfg, name=sbg_rob_lbl)

model_store = store(group="model_cache")
model_store(resnet18_model_cfg, name="resnet18")
model_store(resnet50_model_cfg, name="resnet50")
model_store(simple_cnn_cfg, name="simple_cnn")
model_store(cls_imb_model_s_cfg, name=cls_imb_s_lbl)

params_store = store(group="params_cache")
params_store(dual_optim_params_cfg, name=dual_lbl)
params_store(adam_optim_params_cfg, name="adam")
params_store(single_optim_params_cfg, name=mu_aux_ad_lbl)

schdlr_store = store(group="schdlrs_cache")
schdlr_store(lr_sch_pcfg, name=default_lbl)

grad_accum_schedule_store = store(group="grad_accum_schedule")
grad_accum_schedule_store(grad_accum_schedule_cfg, name=default_lbl)

optims_store = store(group="optims_cache")
optims_store(dual_optims_pcfg, name=dual_lbl)
optims_store(single_adam_pcfg, name="adam")
optims_store(dist_mu_aux_adam_pcfg, name="dist")
optims_store(single_mu_aux_adam_pcfg, name=mu_aux_ad_lbl)

@dataclass
class Experiment_Info:
    name: str
    config_postfix: str
    run_num: int = 0

    def __post_init__(self):
        self.name += f"/{self.config_postfix}"
        global_output_path = Path(f"outputs/{self.name}" )
        all_outputs = sorted([local_path for local_path in os.listdir(global_output_path)])  
        if len(all_outputs) == 0:
            self.output_path = global_output_path / "v_0"
        else:
            last = all_outputs[-1]
            next_i = str(int(last[-1])+1)
            self.output_path = global_output_path / f"v_{next_i}"

        os.chdir(global_output_path)
        self.log_path = Path(f"{self.output_path}/logs/" )

experims_store = store(group="experiment_info")
experims_store(pbuilds(Experiment_Info, "class_imbalance_ratios"), name=cls_imb_s_lbl)
experims_store(pbuilds(Experiment_Info, "spurious_pixel"), name=sp_pix_lbl)
experims_store(pbuilds(Experiment_Info, "subgroup_robustness"), name=sbg_rob_lbl)

logger_store = store(group="logger_cache")
logger_store(csv_logger_cfg, name="csv")
chkpt_store = store(group="chkpt_cache")
chkpt_store(max_checkpt_cfg, name=default_lbl)

# Make a pure dict object like a class, but immediately configurable
ExperimentConfig = make_config(
    seed=1,
    hydra_defaults=[
        {'lit_module': dual_lbl},
        {'params_cache': mu_aux_ad_lbl},
        {'model_cache': "simple_cnn"},
        {'datamodule': sp_pix_lbl},
        {'trainer': default_lbl},        
        {'optims_cache': mu_aux_ad_lbl},
        {'schdlrs_cache': default_lbl},
        {'grad_accum_schedule': default_lbl},
        {'experiment_info': sp_pix_lbl},
        {'logger_cache': "csv"},
        {'chkpt_cache': default_lbl},
        '_self_'
    ]
)

store(ExperimentConfig, name="adam_muon_benchmarks")
pre_seed = zen(lambda seed: pl.seed_everything(seed))
task_function = zen(train_and_val, pre_call=pre_seed)

if __name__ == "__main__":
    store.add_to_hydra_store()
    torch.set_float32_matmul_precision(precision="high")
    
    # TODO: Print yaml config at the end for seeing saved structure
    job = launch(
        ExperimentConfig,
        task_function,
        config_name="adam_muon_benchmarks",
        # job_name="job",
        version_base="1.3",
        multirun=True,
        overrides={
            "model_cache": cls_imb_s_lbl, 
            "datamodule": f"{cls_imb_s_lbl}", 
            "params_cache":f"{mu_aux_ad_lbl}", 
            "optims_cache":f"{mu_aux_ad_lbl}", 
            "lit_module":f"{cls_imb_s_lbl}",
            "++params_cache.lr_muon":f"1e-1",
            "++params_cache.lr_adam":f"1e-1",
            "++params_cache.momentum_muon":f"0.9",
            "++params_cache.beta_1_adam":f"0.95",
            "++params_cache.beta_2_adam":f"0.999",
            "++params_cache.nesterov":f"False",
            "++params_cache.use_layer_recs":f"False",
            "++params_cache.eps":f"1e-08",
            "++datamodule.downsampler_ratio":"20,10,5,2,1" 
        }
    )

