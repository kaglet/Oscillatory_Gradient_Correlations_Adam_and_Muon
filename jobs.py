from hydra_zen import launch, to_yaml
from src.run_w_cfgs import ExperimentConfig, task_function
import torch 
import logging
# Allows jobs based run
if __name__ == "__main__":
    torch.set_float32_matmul_precision(precision="medium")
    logger = logging.getLogger(__name__)
    print(to_yaml(ExperimentConfig))
    job = launch(
        ExperimentConfig,
        task_function,
        config_name="adam_muon_benchmarks",
        
        version_base="1.2",
        overrides=[
            "model_cache=sp_pix", 
            "datamodule=sp_pix", 
            "params_cache=sp_pix", 
            "optims_cache=sp_pix", 
            "lit_module=sp_pix"
        ]
    )