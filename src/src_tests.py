from functools import partial
from hydra_zen import instantiate, just, make_custom_builds_fn
from hydra_zen.third_party.pydantic import validates_with_pydantic
from hydra_zen.third_party.beartype import validates_with_beartype
from beartype.cave import ScalarTypes

from torchvision.models import resnet18, resnet50

class Optimizer:
    def __init__(self, model_weights, learning_rate):
        self.weights = model_weights
        self.lr = learning_rate

def hydra_tests():
    def logger(message: str, format_spec: str) -> str:
        return format_spec.format(message)

    builds = make_custom_builds_fn(
        populate_full_signature=True,
        zen_wrappers=validates_with_pydantic
    )
    pbuilds = make_custom_builds_fn(
        zen_partial=True, 
        populate_full_signature=True,
        
        zen_wrappers=validates_with_beartype
    )

    OptimConf = builds(Optimizer, learning_rate=0.1, model_weights=[1, 2, 3])
    print(OptimConf)

    OptimConf = builds(Optimizer, learning_rate=0.1, zen_partial=True)
    pOptimConf = pbuilds(Optimizer)
    print(pOptimConf)
    # print(pOptimConf([1,2,3]))
    print(instantiate(pOptimConf))
    pOptimFuncToolsObj = instantiate(pOptimConf)
    print(pOptimFuncToolsObj(learning_rate=0.1))

    def f(x: ScalarTypes): return x  # type: ignore # a scalar is any real-valued number
    f([1, 2])
    val_f = validates_with_beartype(f)
    # val_f([1, 2])

    class A:
        def __init__(self, x: ScalarTypes): ... # type: ignore
    validates_with_beartype(A)  # wrapping occurs in-place
    # A([1, 2])

    partial_logger = partial(logger, format_spec='{0:>8s}')
    LogConf = just(partial_logger)
    print(LogConf)
    # partiald_optim = instantiate(OptimConf)
    # print(partiald_optim)

    # partiald_optim(model_weights=[1, 2, 3])

    # partiald_logger = instantiate(LogConf)
    # print(partiald_logger)

    # partiald_logger("hello")
    # print(partiald_logger("goodbye")) 

model50 = resnet50()
print(model50.fc)

model50 = resnet18()
print(model50.fc)