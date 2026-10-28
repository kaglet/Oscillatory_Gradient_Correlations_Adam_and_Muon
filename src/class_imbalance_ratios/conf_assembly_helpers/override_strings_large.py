from pathlib import Path

from src.base.group_name_store import group_names
from src.class_imbalance_ratios.conf_assembly_helpers.group_names import shared_config_names

__all__ = ["adam_overrides", "muon_overrides", "autofill_exceptions"]

cfg_key_downsampler = f"++{group_names.counts_prepper}.downsampler_ratio"
autofill_exceptions = [cfg_key_downsampler]
run_idx = "run_idx"
seed = "seed"
relative_root_raw = str(Path(__file__).parent.parent.parent.parent)
chpc_root_raw = "/mnt/lustre/users/kmotlhabane/Oscillatory_Gradient_Correlations_Adam_and_Muon"
my_root_dir = "my_root_dir"

# TODO: Unify override strings for large with the small case starting with these constants
SMALL_TRAIN_BATCH = 2
SMALL_MAX_EPOCHS = 2
SMALL_NUM_WORKERS = 0
SMALL_PERSISTENT_WORKERS = False
SMALL_LIMIT_TRAIN_BATCHES = 5
SMALL_LIMIT_VAL_BATCHES = 3

NO_LIMIT_TRAIN_BATCHES = "null" #Omega Conf recognised keyword, like False/True, NOT "None"
NO_LIMIT_VAL_BATCHES = "null"

adam_overrides = {
    f"++{run_idx}": "0",
    f"++{seed}": "0",
    f"++{my_root_dir}": relative_root_raw,

    f"{group_names.models}": shared_config_names.experiment, 
    f"{group_names.datamodule}": shared_config_names.experiment, 
    f"{group_names.parameters_set}": shared_config_names.experiment, 
    f"{group_names.optimizers}": shared_config_names.experiment, 
    f"{group_names.lightning_module}": shared_config_names.experiment,
    f"{group_names.checkpoints}": shared_config_names.experiment,
    f"{group_names.trainer}": shared_config_names.experiment,
    f"{group_names.gradient_accumulator_schedule}": shared_config_names.experiment,
    f"{group_names.schedulers}": shared_config_names.adam,

    f"++{group_names.parameters_set}.lr_muon": 1e-1,
    f"++{group_names.parameters_set}.lr_adam": 1e-3,
    f"++{group_names.parameters_set}.use_muon": "False",
    f"++{group_names.parameters_set}.momentum_muon": 0.9,
    f"++{group_names.parameters_set}.beta_1_adam": 0.9,
    f"++{group_names.parameters_set}.beta_2_adam": 0.999,
    f"++{group_names.parameters_set}.nesterov": "False",
    f"++{group_names.parameters_set}.use_layer_recs": "False",
    f"++{group_names.parameters_set}.eps": 1e-8, # Not given
    
    f"{cfg_key_downsampler}": [20, 10, 5, 1], # Part of map blueprint given
    
    f"++{group_names.trainer}.limit_train_batches": SMALL_LIMIT_TRAIN_BATCHES,
    f"++{group_names.trainer}.limit_val_batches": SMALL_LIMIT_VAL_BATCHES,
    f"++{group_names.trainer}.max_epochs": SMALL_MAX_EPOCHS,

    f"++{group_names.datamodule}.num_workers": SMALL_NUM_WORKERS,
    f"++{group_names.datamodule}.persistent_workers": SMALL_PERSISTENT_WORKERS, 
}

muon_overrides = {
    f"++{run_idx}": "0",
    f"++{seed}": "0",
    f"++{my_root_dir}": relative_root_raw,

    f"{group_names.models}": shared_config_names.experiment, 
    f"{group_names.datamodule}": shared_config_names.experiment, 
    f"{group_names.parameters_set}": shared_config_names.experiment, 
    f"{group_names.optimizers}": shared_config_names.experiment, 
    f"{group_names.lightning_module}": shared_config_names.experiment,
    f"{group_names.checkpoints}": shared_config_names.experiment,
    f"{group_names.trainer}": shared_config_names.experiment,
    f"{group_names.gradient_accumulator_schedule}": shared_config_names.experiment,
    f"{group_names.schedulers}": shared_config_names.muon,

    f"++{group_names.parameters_set}.lr_muon": 1e-1, # Use research value
    f"++{group_names.parameters_set}.lr_adam": 3e-4, # Not given - Use default MuonWithAuxAdam rec
    f"++{group_names.parameters_set}.use_muon": "True",
    f"++{group_names.parameters_set}.momentum_muon": 0.9,
    f"++{group_names.parameters_set}.beta_1_adam": 0.9,
    f"++{group_names.parameters_set}.beta_2_adam": 0.95,
    f"++{group_names.parameters_set}.nesterov": "False",
    f"++{group_names.parameters_set}.use_layer_recs": "False",
    f"++{group_names.parameters_set}.eps": 1e-8, # Not given - Use default MuonWithAuxAdam
    
    f"{cfg_key_downsampler}": [20, 10, 5, 1],

    f"++{group_names.trainer}.limit_train_batches": SMALL_LIMIT_TRAIN_BATCHES,
    f"++{group_names.trainer}.limit_val_batches": SMALL_LIMIT_VAL_BATCHES,
    f"++{group_names.trainer}.max_epochs": SMALL_MAX_EPOCHS,

    f"++{group_names.datamodule}.num_workers": SMALL_NUM_WORKERS,
    f"++{group_names.datamodule}.persistent_workers": SMALL_PERSISTENT_WORKERS, 
}

if __name__ == "__main__":
    # Most important check
    print(f"Relative root is: {relative_root_raw}")