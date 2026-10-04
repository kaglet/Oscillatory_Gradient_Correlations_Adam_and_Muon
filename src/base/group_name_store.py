"""
"Cache" labels denote groups that can contain partially built i.e. wrapped objects. 
functools.partial can be used as an example to achieve this. 
They are this way as their true value is only known at run time, 
depending on other objects that can be specified upfront. 
"""

from dataclasses import dataclass
from types import SimpleNamespace

__all__ = ["group_names"]

@dataclass
class GroupNameStore:
    lightning_module:str="lit_module"
    trainer="trainer"
    datamodule = "datamodule"
    models = "model_cache"
    parameters_set = "params_cache"
    schedulers = "schdlrs_cache"
    gradient_accumulator_schedule = "grad_accum_schedule"
    optimizers = "optims_cache"
    loggers = "logger_cache"
    checkpoints = "chkpt_cache"  
    overrides_manager = "overrides_run_manager"   

group_names = GroupNameStore()