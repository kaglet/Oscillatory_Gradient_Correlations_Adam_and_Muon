from hydra_zen import builds, instantiate
from pytorch_lightning import Trainer

trainer_cfg = builds(Trainer)
trainer = instantiate(trainer_cfg)
print(trainer)