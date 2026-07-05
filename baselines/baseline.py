from hydra.utils import instantiate, target_whitelist

with target_whitelist("main.*"):
    with target_whitelist("main.models.*"):
        with target_whitelist("main.optimizers.*"):
            train(cfg)