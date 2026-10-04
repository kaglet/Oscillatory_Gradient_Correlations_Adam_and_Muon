from src.base.group_name_store import group_names
from src.class_imbalance_ratios.group_names import shared_config_names

__all__ = ["adam_overrides", "muon_overrides", "autofill_exceptions"]

cfg_key_downsampler = f"++{group_names.datamodule}.downsampler_ratios"
autofill_exceptions = [cfg_key_downsampler]

adam_overrides = {
    f"{group_names.models}": shared_config_names.experiment, 
    f"{group_names.datamodule}": shared_config_names.experiment, 
    f"{group_names.parameters_set}": shared_config_names.experiment, 
    f"{group_names.optimizers}": shared_config_names.experiment, 
    f"{group_names.lightning_module}": shared_config_names.experiment,
    f"{group_names.overrides_manager}": shared_config_names.experiment,
    f"++{group_names.parameters_set}.lr_muon": 1e-1,
    f"++{group_names.parameters_set}.lr_adam": 1e-1,
    f"++{group_names.parameters_set}.use_muon": "False",
    f"++{group_names.parameters_set}.momentum_muon": 0.9,
    f"++{group_names.parameters_set}.beta_1_adam": 0.95,
    f"++{group_names.parameters_set}.beta_2_adam": 0.999,
    f"++{group_names.parameters_set}.nesterov": "False",
    f"++{group_names.parameters_set}.use_layer_recs": "False",
    f"++{group_names.parameters_set}.eps": 1e-10,
    f"{cfg_key_downsampler}":"[20, 10, 5, 1]",
    f"++{group_names.trainer}.limit_val_batches": 200,
    f"++{group_names.trainer}.max_epochs": 50,
}

muon_overrides = {
    f"{group_names.models}": shared_config_names.experiment, 
    f"{group_names.datamodule}": shared_config_names.experiment, 
    f"{group_names.parameters_set}": shared_config_names.experiment, 
    f"{group_names.optimizers}": shared_config_names.experiment, 
    f"{group_names.lightning_module}": shared_config_names.experiment,
    f"++{group_names.parameters_set}.lr_muon": 1e-1,
    f"++{group_names.parameters_set}.lr_adam": 1e-1,
    f"++{group_names.parameters_set}.use_muon": "False",
    f"++{group_names.parameters_set}.momentum_muon": 0.9,
    f"++{group_names.parameters_set}.beta_1_adam": 0.95,
    f"++{group_names.parameters_set}.beta_2_adam": 0.999,
    f"++{group_names.parameters_set}.nesterov": "False",
    f"++{group_names.parameters_set}.use_layer_recs": "False",
    f"++{group_names.parameters_set}.eps": 1e-10,
    f"{cfg_key_downsampler}":"[20, 10, 5, 1]",
    f"++{group_names.trainer}.limit_val_batches": 200,
    f"++{group_names.trainer}.max_epochs": 50,
}