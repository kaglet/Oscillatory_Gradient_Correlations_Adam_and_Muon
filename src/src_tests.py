from functools import partial
from hydra_zen import instantiate, just, make_custom_builds_fn
from hydra_zen.third_party.pydantic import validates_with_pydantic

class Optimizer:
    def __init__(self, model_weights, learning_rate):
        self.weights = model_weights
        self.lr = learning_rate

def logger(message: str, format_spec: str) -> str:
    return format_spec.format(message)

builds = make_custom_builds_fn(
    populate_full_signature=True,
    zen_wrappers=validates_with_pydantic
)
pbuilds = make_custom_builds_fn(
    zen_partial=True, 
    populate_full_signature=True,
)
OptimConf = builds(Optimizer, learning_rate=0.1, zen_partial=True)

partial_logger = pbuilds(logger, format_spec='{0:>8s}')
LogConf = just(partial_logger)

partiald_optim = instantiate(OptimConf)
print(partiald_optim)

partiald_optim(model_weights=[1, 2, 3])

partiald_logger = instantiate(LogConf)
print(partiald_logger)

partiald_logger("hello")
print(partiald_logger("goodbye")) 