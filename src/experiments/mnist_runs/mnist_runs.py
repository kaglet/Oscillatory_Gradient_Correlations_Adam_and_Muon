from math import pi

import pytorch_lightning as pl
from hydra_zen import builds, make_config, make_custom_builds_fn, zen
import torch as tr
from torch.optim import Adam  # type: ignore
from torch.utils.data import DataLoader
from src.trainer.modules.mnist_zen_module import MNISTLitModule 
from src.trainer.dataloader import mnist_zen_dataloader
from src.trainer.per_optim_params.transformer_params import get_params
from src import train_and_eval

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)

# TODO: FIX passing everything into here instead of already built trainer and lit_module
# See file mnist_trainer.py where I have matching note and instructions, that is the task_eval function to adopt
ExperimentConfig = make_config(
    seed=1,
    lit_module=MNISTLitModule,
    trainer=builds(pl.Trainer, max_epochs=100),
    model=builds(, num_neurons=10),
    optim=pbuilds(Adam),
    dataloader=pbuilds(DataLoader, batch_size=25, shuffle=True, drop_last=True),
    target_fn=tr.cos,
    training_domain=builds(tr.linspace, start=-2 * pi, end=2 * pi, steps=1000),
)

# Wrapping `train_and_eval` with `zen` makes it compatible with Hydra as a task function
#
# We must specify `pre_call` to ensure that pytorch lightning seeds everything
# *before* any of our configs are instantiated (which will initialize the pytorch
# model whose weights depend on the seed)
pre_seed = zen(lambda seed: pl.seed_everything(seed))
task_function = zen(train_and_eval, pre_call=pre_seed)

if __name__ == "__main__":
    # enables us to call
    from hydra_zen import ZenStore

    store = ZenStore(deferred_hydra_store=False)
    store(ExperimentConfig, name="lit_app")

    task_function.hydra_main(
        config_name="lit_app",
        version_base="1.1",
        config_path=".",
    )