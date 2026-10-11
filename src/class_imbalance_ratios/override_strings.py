from src.base.group_name_store import group_names
from src.class_imbalance_ratios.group_names import shared_config_names

__all__ = ["adam_overrides", "muon_overrides", "autofill_exceptions"]

cfg_key_downsampler = f"++{group_names.counts_prepper}.downsampler_ratio"
autofill_exceptions = [cfg_key_downsampler]
run_idx = "run_idx"
seed = "seed"

adam_overrides = {
    f"++{run_idx}": "0",
    f"++{seed}": "0",

    f"{group_names.models}": shared_config_names.experiment, 
    f"{group_names.datamodule}": shared_config_names.experiment, 
    f"{group_names.parameters_set}": shared_config_names.experiment, 
    f"{group_names.optimizers}": shared_config_names.experiment, 
    f"{group_names.lightning_module}": shared_config_names.experiment,
    f"{group_names.checkpoints}": shared_config_names.experiment,
    f"{group_names.trainer}": shared_config_names.experiment,
    f"{group_names.gradient_accumulator_schedule}": shared_config_names.experiment,
    f"{group_names.schedulers}": shared_config_names.adam,
    f"{group_names.my_root_dir}": shared_config_names.local,

    f"++{group_names.parameters_set}.lr_muon": 1e-1,
    f"++{group_names.parameters_set}.lr_adam": 1e-3,
    f"++{group_names.parameters_set}.use_muon": "False",
    f"++{group_names.parameters_set}.momentum_muon": 0.9,
    f"++{group_names.parameters_set}.beta_1_adam": 0.9,
    f"++{group_names.parameters_set}.beta_2_adam": 0.999,
    f"++{group_names.parameters_set}.nesterov": "False",
    f"++{group_names.parameters_set}.use_layer_recs": "False",
    f"++{group_names.parameters_set}.eps": 1e-10, # Not given
    
    f"{cfg_key_downsampler}": [20, 10, 5, 1], # Part of map blueprint given
    
    f"++{group_names.trainer}.limit_train_batches": 10,
    f"++{group_names.trainer}.limit_val_batches": 3,
    f"++{group_names.trainer}.max_epochs": 2,

    f"++{group_names.datamodule}.num_workers": 0, 
    f"++{group_names.datamodule}.persistent_workers": "False",
}

muon_overrides = {
    f"++{run_idx}": "0",
    f"++{seed}": "0",

    f"{group_names.models}": shared_config_names.experiment, 
    f"{group_names.datamodule}": shared_config_names.experiment, 
    f"{group_names.parameters_set}": shared_config_names.experiment, 
    f"{group_names.optimizers}": shared_config_names.experiment, 
    f"{group_names.lightning_module}": shared_config_names.experiment,
    f"{group_names.checkpoints}": shared_config_names.experiment,
    f"{group_names.trainer}": shared_config_names.experiment,
    f"{group_names.gradient_accumulator_schedule}": shared_config_names.experiment,
    f"{group_names.schedulers}": shared_config_names.muon,
    f"{group_names.my_root_dir}": shared_config_names.local,

    f"++{group_names.parameters_set}.lr_muon": 1e-1,
    f"++{group_names.parameters_set}.lr_adam": 1e-3, # Not given
    f"++{group_names.parameters_set}.use_muon": "True",
    f"++{group_names.parameters_set}.momentum_muon": 0.9,
    f"++{group_names.parameters_set}.beta_1_adam": 0.9,
    f"++{group_names.parameters_set}.beta_2_adam": 0.999,
    f"++{group_names.parameters_set}.nesterov": "False",
    f"++{group_names.parameters_set}.use_layer_recs": "False",
    f"++{group_names.parameters_set}.eps": 1e-10, # Not given
    
    f"{cfg_key_downsampler}": [20, 10, 5, 1],

    f"++{group_names.trainer}.limit_train_batches": 10,
    f"++{group_names.trainer}.limit_val_batches": 3,
    f"++{group_names.trainer}.max_epochs": 2,

    f"++{group_names.datamodule}.num_workers": 0, 
    f"++{group_names.datamodule}.persistent_workers": "False", 
}