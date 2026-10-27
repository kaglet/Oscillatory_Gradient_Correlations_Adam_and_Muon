
"""
Anything configurable or static in a configurable aspect is defined here and assembled later.
"""

import argparse
from dataclasses import dataclass
import logging
import math
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ConstantLR
from hydra.conf import HydraConf, JobConf, SweepDir, RunDir

from src.base.group_name_store import group_names
from src.base.lit_module.components.callbacks.custom_callbacks import DebugCallback, StoreAllGCH, TimerCallback
from src.class_imbalance_ratios.datamodule.cls_imb_datamodule import ClsImbDataModule, ClsImbSmallDM
from src.base.lit_module.components.grad_accum.custom_grad_accum import GradAccumSchedule
from src.class_imbalance_ratios.lit_module.components.custom_resnet import create_resnet_model
from src.base.lit_module.components.per_optim_params.resnet_params.get_params import get_nondist_Muon_Adam_params
from src.base.lit_module.general_lit_module import DualOptLitModule
from src.class_imbalance_ratios.conf_assembly_helpers.artificial_overrides_padder import ArtificialOverridesPadder
from src.class_imbalance_ratios.datamodule.components.ratio_prepper import CountsPrepperFromRatio
from src.class_imbalance_ratios.conf_assembly_helpers.group_names import shared_config_names
from src.class_imbalance_ratios.conf_assembly_helpers.override_strings import adam_overrides, muon_overrides, autofill_exceptions
from src.spurious_pix.datamodule.components.pixel_injection_map import PixelInjectionMap
from src.spurious_pix.lit_module.components.simple_net import SpPixCNN
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
model_cfg = builds(SpPixCNN)

# Optimiser variations
mu_aux_adam_pcfg = [pbuilds(SingleDeviceMuonWithAuxAdam)]

# Gradient accum schedules 
grad_accum_schedule_cfg = builds(GradAccumSchedule, {0: 1})

# LR Scheduler
adam_lr_sch_pcfg = pbuilds(ConstantLR, factor=1)
muon_lr_sch_pcfg = pbuilds(ConstantLR, factor=1)

single_lit_module_cfg = pbuilds(DualOptLitModule)

class_labels = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine"
]
num_classes = len(class_labels)

cls_imb_s_module_cfg = pbuilds(
    DualOptLitModule,
    grad_accum_schedule = grad_accum_schedule_cfg,
    num_classes = num_classes,
    labels=class_labels
)

# ---- TRAINER CONFIGS START ---- #
trainer_dir = "${my_root_dir}"

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
    max_epochs=10,
    deterministic=True,
    benchmark=False,
    detect_anomaly=False,
    devices=1, 
    enable_checkpointing = False,
    callbacks=[builds(StoreAllGCH), builds(DebugCallback), builds(TimerCallback)]
)  

# ---- DATAMODULE CONFIGS START ---- #
horizontal_flip_cfg = builds(torchvision.transforms.RandomHorizontalFlip)
crops_cfg = builds(torchvision.transforms.RandomCrop)
to_tensor_cfg = builds(torchvision.transforms.ToTensor)

IMG_WIDTH = 32
transforms_cfg = builds(
    torchvision.transforms.Compose,
    [
        horizontal_flip_cfg,
        crops_cfg(size=IMG_WIDTH, padding=4), # Above transforms act on images not tensors, hence to tensor is last
        to_tensor_cfg,
    ]
)

labels = list(range(num_classes))
train_class_counts = [4200 for _ in range(num_classes)]
train_map = {k:v for k,v in zip(labels, train_class_counts)}

val_class_counts = [1800 for _ in range(num_classes)]
val_map = {k:v for k,v in zip(labels, val_class_counts)}

test_class_counts = [0 for _ in range(num_classes)]
test_map = {k:v for k,v in zip(labels, test_class_counts)}

DOWNSAMPLER_RATIO = 1
pixel_map_cfg = builds(PixelInjectionMap, labels=labels, width_only=IMG_WIDTH, injection_seed="${injection_seed}")
counts_prepper_cfg = builds(CountsPrepperFromRatio, ratio_seed="${ratio_seed}", downsampler_ratio=DOWNSAMPLER_RATIO, train_map=train_map, val_map=val_map, test_map=test_map)

runtime_experim_dir = "${my_root_dir}/outputs/${hydra.job.name}/${hydra.job.config_name}/multijob/R_${counts_prepper.downsampler_ratio}"
runtime_run_subdir = "${hydra:runtime.output_dir}/run_${run_idx}/" # This path is relative to cwd (current working dir) set by Hydra during program run, could use an interpolated function

data_dir = "${my_root_dir}/data/"
# print("data_dir set to: ", data_dir)
cls_imb_datamdl_s_cfg = builds(
    ClsImbSmallDM, 
    persistent_workers=True,
    data_dir = data_dir,
    train_transforms = transforms_cfg,
    val_transforms = transforms_cfg,
    test_transforms = transforms_cfg,
    batch_size = 512, # From batch size set grad_accum to get new effective batch size
    num_workers = 3,
    metadata_dir = runtime_run_subdir,
    labels=class_labels,
    counts_prepper=counts_prepper_cfg,
)

lit_module_store = store(group=group_names.lightning_module)
lit_module_store(cls_imb_s_module_cfg, name=shared_config_names.experiment)

trainer_store = store(group=group_names.trainer)
trainer_store(trainer_cfg, name=shared_config_names.experiment)

datamodule_store = store(group=group_names.datamodule)
datamodule_store(cls_imb_datamdl_s_cfg, name=shared_config_names.experiment)

model_store = store(group=group_names.models)
model_store(model_cfg, name=shared_config_names.experiment)

params_store = store(group=group_names.parameters_set)
params_store(mu_aux_adam_params_cfg, name=shared_config_names.experiment)

schdlr_store = store(group=group_names.schedulers)
schdlr_store(adam_lr_sch_pcfg, name=shared_config_names.adam)
schdlr_store(muon_lr_sch_pcfg, name=shared_config_names.muon)

grad_accum_schedule_store = store(group=group_names.gradient_accumulator_schedule)
grad_accum_schedule_store(grad_accum_schedule_cfg, name=shared_config_names.experiment)

optims_store = store(group=group_names.optimizers)
optims_store(mu_aux_adam_pcfg, name=shared_config_names.experiment)

logger_store = store(group=group_names.loggers)
logger_store(csv_logger_cfg, name=shared_config_names.experiment)

chkpt_store = store(group=group_names.checkpoints)
chkpt_store(max_checkpt_cfg, name=shared_config_names.experiment)

counts_prepper_store = store(group=group_names.counts_prepper)
counts_prepper_store(counts_prepper_cfg, name=shared_config_names.experiment)

# Make a pure dict object like a class, but immediately configurable
ExperimentConfig = make_config(
    seed=1, # TODO: Make new seed per run as param to overrides not here 
    hydra_defaults=[
        {group_names.lightning_module: shared_config_names.experiment},

        {group_names.parameters_set: shared_config_names.experiment},

        {group_names.models: shared_config_names.experiment},

        {group_names.datamodule: shared_config_names.experiment},

        {group_names.trainer: shared_config_names.experiment},      

        {group_names.optimizers: shared_config_names.experiment},

        {group_names.schedulers: shared_config_names.muon},

        {group_names.gradient_accumulator_schedule: shared_config_names.experiment},

        {group_names.loggers: shared_config_names.experiment},

        {group_names.checkpoints: shared_config_names.experiment},

        {group_names.counts_prepper: shared_config_names.experiment},

        '_self_'
    ]
)

pre_seed = zen(lambda seed: pl.seed_everything(seed))
task_function = zen(train_and_val, pre_call=pre_seed) # Remove seeding here caught as input to config

store(
    HydraConf(
        job=JobConf(chdir=True), 
        run=RunDir(runtime_experim_dir),
        output_subdir="run_${run_idx}/hydra",  # TODO: Ensure run_idx is interpolated right with new override structure
    ),
    group="hydra",
    name="config"
)  # store `hydra.job.chdir=True`

store.add_to_hydra_store(overwrite_ok=True)

def pyaml(x):
    # for pretty printing configs
    print(to_yaml(x))

# pyaml(store["hydra", "config"])

if __name__ == "__main__":
    """ 
    Via CLI. 
    Fix and pair run/seed (same value for ease) 
    Then slice through the current experiment matches. 
    Here there are 4 experiment ratios to test. 
    """
    log_file = Path(__file__).parent / "logs_run_with_cfg.log"
    logging.basicConfig(filename=log_file, level=logging.DEBUG)

    parser = argparse.ArgumentParser(
        description="Script that adds 3 numbers from CMD"
    )
    parser.add_argument("--run", required=False, nargs="?", type=int, help="Enter `run`", default=0)
    parser.add_argument("--seed", required=False, nargs="?", type=int, help="Enter `seed`", default=0)
    parser.add_argument("--slice", required=False, nargs="?", type=int, help="Enter `slice`", default=0)
    parser.add_argument("--ratio_seed", required=False, nargs="?", type=int, help="Enter `run_seed`", default=0)

    args = parser.parse_args()

    injected_run = args.run
    injected_seed = args.seed
    injected_slice = args.slice
    injected_ratio_seed = args.ratio_seed

    muon_overrides_map = ArtificialOverridesPadder(muon_overrides, autofill_exceptions, 4).overrides_map_as_store
    muon_overrides_map.overwrite(
        "++run_idx", 
        [f"{injected_run}" for _ in list(range(4))]
    )
    muon_overrides_map.overwrite(
        "++seed",
        [f"{injected_seed}" for _ in list(range(4))]
    )
    muon_overrides_map.overwrite(
        "++ratio_seed",
        [f"{injected_ratio_seed}" for _ in list(range(4))]
    )
    curr_vals = muon_overrides_map.get_map_slice(injected_slice)

    job = launch(
        ExperimentConfig,
        task_function,
        config_name="muon_class_imbalance_ratios",
        job_name="cls_imb_s_tests",
        version_base="1.3",
        overrides=curr_vals
    )

    adam_overrides_map = ArtificialOverridesPadder(adam_overrides, autofill_exceptions, 4).overrides_map_as_store

    adam_overrides_map.overwrite(
        "run_idx", 
        [f"{injected_run}" for _ in list(range(4))]
    )
    adam_overrides_map.overwrite(
        "seed",
        [f"{injected_seed}" for _ in list(range(4))]
    )
    adam_overrides_map.overwrite(
        "++ratio_seed",
        [f"{injected_ratio_seed}" for _ in list(range(4))]
    )
    curr_vals = adam_overrides_map.get_map_slice(injected_slice)

    job = launch(
        ExperimentConfig,
        task_function,
        config_name="adam_class_imbalance_ratios",
        job_name="cls_imb_s_tests",
        version_base="1.3",
        overrides=curr_vals
    )
