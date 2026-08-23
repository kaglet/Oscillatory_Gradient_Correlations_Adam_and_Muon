from hydra_zen import launch, instantiate
from src.hydra_plugins.resolvers import register_resolvers
from src.run_w_cfgs import ExperimentConfig, task_function
import torch 

# Allows jobs based run
if __name__ == "__main__":
    # register_resolvers()
    torch.set_float32_matmul_precision(precision="medium")
    job = launch(
        ExperimentConfig,
        task_function,
        config_name="adam_muon_benchmarks",
        version_base="1.3"
    )