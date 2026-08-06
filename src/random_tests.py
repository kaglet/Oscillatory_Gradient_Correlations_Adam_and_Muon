from hydra_zen import builds
from pytorch_lightning import Trainer

trainer_cfg = builds(Trainer)
print(trainer_cfg)