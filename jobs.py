from hydra_zen import launch, instantiate
from src.run_w_cfgs import ExperimentConfig, task_function
import torch

if __name__ == "__main__":
    torch.set_warn_always(False)
    job = launch(
        ExperimentConfig,
        task_function,
        config_name="lit_app",
        version_base="1.2"
    )