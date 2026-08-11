from hydra_zen import launch
from cos700.src.run_w_cfgs import ExperimentConfig, task_function

(jobs,) = launch(
    ExperimentConfig,
    task_function,
    multirun=True,
)