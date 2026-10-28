import math
import numpy as np
import torch

class PixelInjectionMap():
    def __init__(self, labels: list[int], width_only: int, injection_seed: int) -> None:
        rng = np.random.default_rng(seed=injection_seed)
        self._injections_mapping = {}
        for lbl in labels:
            lbl = str(lbl)
            injection_site = rng.choice(np.arange(width_only), size=2, replace=False)
            self._injections_mapping[lbl] = list(injection_site)

    @property
    def injections_mapping(self) -> dict[str, list[int]]:
        return self._injections_mapping

if __name__ == "__main__":
    num_classes = 10
    labels = list(range(num_classes))

    injector = PixelInjectionMap(labels, 32, 1)
    print(injector.injections_mapping)

