
"""
Anything configurable or static in a configurable aspect is defined here and assembled later.
"""

from dataclasses import dataclass
import os
import multiprocessing
from pathlib import Path
from typing import Any, Literal, Type

from omegaconf import OmegaConf
import torch
import torchvision
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from hydra_zen import instantiate, to_yaml, just, launch, make_custom_builds_fn, make_config, make_custom_builds_fn, multirun, zen, store
from muon import MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from torch.optim import Muon, Adam
from torch.optim.lr_scheduler import LRScheduler
from hydra.conf import HydraConf, JobConf, SweepDir, RunDir

from src.base.configs.experiment_data import Experiment_Data
from src.base.datamodule.cifar_datamodule import CIFARDataModule
from src.base.lit_module.components.custom_callbacks import DebugCallback, StoreAllGCH
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
from hydra_zen.third_party.pydantic import validates_with_pydantic
# from pl_bolts.transforms.dataset_normalizations import cifar10_normalization, imagenet_normalization

builds = make_custom_builds_fn(populate_full_signature=True)
pbuilds = make_custom_builds_fn(
    zen_partial=True, 
    populate_full_signature=True
)

# ---- LIGHTNING MODULE CONFIGS BEGIN ---- #
# Define base params cfg
mu_aux_adam_params_cfg = pbuilds(
    get_nondist_Muon_Adam_params,
)

# Define base vision model cfg
resnet18_model_cfg = builds(create_resnet_model, net=just(resnet18))
resnet50_model_cfg = builds(create_resnet_model, net=just(resnet50))

cls_imb_s_num_classes = 10
cls_imb_l_num_classes = 100

cls_imb_model_s_cfg = resnet18_model_cfg(num_classes=cls_imb_s_num_classes, num_in=3)
cls_imb_model_l_cfg = resnet50_model_cfg(num_classes=cls_imb_l_num_classes, num_in=3) 

# Optimiser variations
mu_aux_adam_pcfg = [pbuilds(SingleDeviceMuonWithAuxAdam)]

# Gradient accum schedules 
grad_accum_schedule_cfg = builds(GradAccumSchedule, {0: 32})

# LR Schedulers
lr_sch_pcfg = [pbuilds(LRScheduler), pbuilds(LRScheduler)]

single_lit_module_cfg = pbuilds(SingleOptLitModule)

# Lighning Module variations (for groups mainly for swap in and out, TODO: Defer, ideas below)
# Top level DualOptLit module cannot be changed, the top can be registered or even try dictionary keys to unpack as keyword arguments
cls_imb_s_module_cfg = pbuilds(
    SingleOptLitModule,
    grad_accum_schedule = grad_accum_schedule_cfg,
    num_classes = cls_imb_s_num_classes
)

cls_imb_l_module_cfg = pbuilds(
    SingleOptLitModule, 
    grad_accum_schedule = grad_accum_schedule_cfg,
    num_classes = cls_imb_l_num_classes
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

tb_logger_cfg = pbuilds(TensorBoardLogger, name="train")
csv_logger_cfg = [
    pbuilds(CSVLogger, name="train", version=""), # Remove version nesting
    pbuilds(CSVLogger, name="val", version=""), 
    pbuilds(CSVLogger, name="test", version=""), 
]

# Define base trainer cfg
trainer_cfg = pbuilds(
    pl.Trainer, 
    num_sanity_val_steps = 0,
    max_epochs=2,
    deterministic=True,
    accelerator="gpu",
    benchmark=False,
    detect_anomaly=False,
    devices=1, 
    # precision = "bf16-mixed",
    callbacks=[builds(StoreAllGCH), builds(DebugCallback)]
)  

# ---- DATAMODULE CONFIGS START ---- #
horizontal_flip_cfg = builds(torchvision.transforms.RandomHorizontalFlip)
crops_cfg = builds(torchvision.transforms.RandomCrop)
to_tensor_cfg = builds(torchvision.transforms.ToTensor)

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

total_local_cores = 0
total_local_cores = total_local_cores if type(total_local_cores) == int else 1

cls_imb_datamdl_s_cfg = builds(
    ClsImbDataModule, 
    train_transforms = cls_imb_s_transforms,
    val_transforms = cls_imb_s_transforms,
    test_transforms = cls_imb_s_transforms,
    batch_size = 64, # From batch size set grad_accum to get new effective batch size
    num_workers = 0
)

cls_imb_datamdl_l_cfg = builds(
    ClsImbDataModule, 
    downsampler_ratio=20, 
    weighted_ratios=None,
    train_transforms = cls_imb_l_transforms,
    val_transforms = cls_imb_l_transforms,
    test_transforms = cls_imb_l_transforms,
    batch_size = 64,
    num_workers = 0
)

# TODO: Store these overlapping keys in a dictionary rather to not keep repeating prefix values and to make config easier
mu_aux_ad_lbl = "mu_aux_ad"
adam_lbl = "adam"
task_lbl = "cls_imb"
cif10_lbl = f"cif10"
cif100_lbl = f"cif100"

# Create groups: Key value pairs (At current granularity level desired to register options per group)
lit_module_store = store(group="lit_module")
lit_module_store(cls_imb_s_module_cfg, name=cif10_lbl)
lit_module_store(cls_imb_l_module_cfg, name=cif100_lbl)

trainer_store = store(group="trainer")
trainer_store(trainer_cfg, name=task_lbl)

datamodule_store = store(group="datamodule")
datamodule_store(cls_imb_datamdl_s_cfg, name=cif10_lbl)
datamodule_store(cls_imb_datamdl_l_cfg, name=cif100_lbl)

model_store = store(group="model_cache")
model_store(cls_imb_model_s_cfg, name=cif10_lbl)
model_store(cls_imb_model_l_cfg, name=cif100_lbl)

params_store = store(group="params_cache")
params_store(mu_aux_adam_params_cfg, name=mu_aux_ad_lbl)

schdlr_store = store(group="schdlrs_cache")
schdlr_store(lr_sch_pcfg, name=task_lbl)

grad_accum_schedule_store = store(group="grad_accum_schedule")
grad_accum_schedule_store(grad_accum_schedule_cfg, name=task_lbl)

optims_store = store(group="optims_cache")
optims_store(mu_aux_adam_pcfg, name=mu_aux_ad_lbl)

experims_store = store(group="experiment_info")
experims_store(pbuilds(Experiment_Data, "class_imbalance_ratios"), name=task_lbl)

logger_store = store(group="logger_cache")
logger_store(csv_logger_cfg, name=task_lbl)

chkpt_store = store(group="chkpt_cache")
chkpt_store(max_checkpt_cfg, name=task_lbl)

# Make a pure dict object like a class, but immediately configurable
ExperimentConfig = make_config(
    seed=1,
    hydra_defaults=[
        {'lit_module': cif10_lbl},
        {'params_cache': mu_aux_ad_lbl},
        {'model_cache': cif10_lbl},
        {'datamodule': cif10_lbl},
        {'trainer': task_lbl},        
        {'optims_cache': mu_aux_ad_lbl},
        {'schdlrs_cache': task_lbl},
        {'grad_accum_schedule': task_lbl},
        {'experiment_info': task_lbl},
        {'logger_cache': task_lbl},
        {'chkpt_cache': task_lbl},
        '_self_'
    ]
)

store(ExperimentConfig, name="adam_muon_benchmarks", group="experiments")
pre_seed = zen(lambda seed: pl.seed_everything(seed))
task_function = zen(train_and_val, pre_call=pre_seed)

store.add_to_hydra_store(overwrite_ok=True)

store(
    HydraConf(
        job=JobConf(chdir=True), 
        run=RunDir("outputs/${hydra.job.name}/multijob/R_${datamodule.downsampler_ratio}"),
        output_subdir="hydra_run_${run_idx}", 
    ),
    group="hydra",
    name="config"
)  # store `hydra.job.chdir=True`

store.add_to_hydra_store(overwrite_ok=True)

# cfg_2_yaml = to_yaml(store["hydra", "config"])

# TODO: Move this, depends on cif10 labels and so on, which should be in a separate file
class UnifiedJobsStore():
    """ 
    Unified serial job run store for Muon and Adam. Can later be separated as potential TODO. 
    This stores values for a series of experiments. It does not control running experiments multiple times at a meta level. 
    The `get_run_vals` function augments this functional need however. 
    """
    def __init__(self, downsampler_ratios: list[int] = [20, 10, 5, 1]) -> None:
        self._downsampler_ratios = downsampler_ratios
        self._job_num_each = len(self._downsampler_ratios)
        self._adam_overrides = self._set_adam_overrides()
        self._muon_overrides = self._set_muon_overrides()

    def _repeat_join_str(self, val: str | int | float) -> list[str]:
        val = str(val) if type(val) is int else val
        string = [f"{val}"]
        string *= self._job_num_each
        string = string
        return string   

    def _set_muon_overrides(self): 
        adam_overrides = {
            "model_cache": self._repeat_join_str(cif10_lbl), 
            "datamodule": self._repeat_join_str(cif10_lbl), 
            "params_cache": self._repeat_join_str(mu_aux_ad_lbl), 
            "optims_cache": self._repeat_join_str(mu_aux_ad_lbl), 
            "lit_module": self._repeat_join_str(cif10_lbl),
            "++run_idx": self._repeat_join_str(2),
            "++params_cache.lr_muon": self._repeat_join_str(1e-1),
            "++params_cache.lr_adam": self._repeat_join_str(1e-1),
            "++params_cache.momentum_muon": self._repeat_join_str(0.9),
            "++params_cache.beta_1_adam": self._repeat_join_str(0.95),
            "++params_cache.beta_2_adam": self._repeat_join_str(0.999),
            "++params_cache.nesterov": self._repeat_join_str("False"),
            "++params_cache.use_layer_recs": self._repeat_join_str("False"),
            "++params_cache.eps": self._repeat_join_str(1e-8),
            "++datamodule.downsampler_ratio":self.downsampler_ratios,
            "++trainer.limit_val_batches": self._repeat_join_str(6000),
        }

        return adam_overrides

    def _set_adam_overrides(self): # TODO: Find way to merge into each dict intelligently shared definite params to not repeat. Could do merge dict after defining shared properties.
        muon_overrides = {
        "model_cache":self._repeat_join_str(cif10_lbl), 
        "datamodule":self._repeat_join_str(cif10_lbl), 
        "params_cache":self._repeat_join_str(mu_aux_ad_lbl), 
        "optims_cache":self._repeat_join_str(mu_aux_ad_lbl), 
        "lit_module":self._repeat_join_str(cif10_lbl),
        "++run_idx": self._repeat_join_str(2),
        "++params_cache.lr_muon":self._repeat_join_str(1e-1),
        "++params_cache.lr_adam":self._repeat_join_str(1e-1),
        "++params_cache.use_muon":self._repeat_join_str("False"),
        "++params_cache.momentum_muon":self._repeat_join_str(0.9),
        "++params_cache.beta_1_adam":self._repeat_join_str(0.95),
        "++params_cache.beta_2_adam":self._repeat_join_str(0.999),
        "++params_cache.nesterov":self._repeat_join_str("False"),
        "++params_cache.use_layer_recs":self._repeat_join_str("False"),
        "++params_cache.eps":self._repeat_join_str(1e-8),
        "++datamodule.downsampler_ratio":self.downsampler_ratios,
        "++trainer.limit_val_batches": self._repeat_join_str(6000),
        }

        return muon_overrides
    
    def get_run_vals(self, outer_run: int, inner_experim_idx: int = 0, which: Literal['muon', "adam"] = 'muon') -> dict[str, str]:
        """ 
        
        """
        overrides = self._muon_overrides if which == "muon" else self._adam_overrides
        valid_slice = {k:v[inner_experim_idx] for k,v in overrides.items()}
        run_slice = {**valid_slice, "++run_idx":f"{str(outer_run)}"}
        return run_slice

    @property
    def job_num_each(self):
        return self._job_num_each

    @property
    def downsampler_ratios(self):
        return self._downsampler_ratios

if __name__ == "__main__":
    """ 
    Now emergent structure is 1 run idx, shared for Muon and Adam.
    Do all inner experiments for 1 run, Muon for a ratio, then Adam. 
    """
    jobs_store = UnifiedJobsStore()
    run_idxs = [0,1,2,3]
    for r in run_idxs:
        muon_counter = 0
        adam_counter = 0
        for i in range(jobs_store.job_num_each * 2):
            if i % 2 == 0:
                # TODO: Make experims idx based too, dynamically merged, or set maybe idk
                curr_job = jobs_store.get_run_vals(r, inner_experim_idx=muon_counter, which='muon')
                job = launch(
                    ExperimentConfig,
                    task_function,
                    config_name="adam_muon_benchmarks",
                    job_name="muon_class_imbalance_ratios",
                    version_base="1.3",
                    overrides=curr_job
                )
                muon_counter += 1
            else:
                curr_job = jobs_store.get_run_vals(r, inner_experim_idx=adam_counter, which='adam')
                job = launch(
                    ExperimentConfig,
                    task_function,
                    config_name="adam_muon_benchmarks",
                    job_name="adam_class_imbalance_ratios",
                    version_base="1.3",
                    overrides=curr_job
                )
                adam_counter += 1

