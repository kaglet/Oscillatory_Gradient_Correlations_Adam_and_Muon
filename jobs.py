from hydra_zen import launch, instantiate
from src.run_w_cfgs import ExperimentConfig, task_function

job = launch(
    ExperimentConfig,
    task_function,
    config_name="lit_app",
    version_base="1.2"
)