from hydra_zen import zen, store
from hydra.conf import HydraConf, JobConf, RunDir, OverridesConf

def task():  # just an example task function - no important details
    import os
    print(f"working dir: {os.getcwd()}")

if __name__ == "__main__":
    store(HydraConf(job=JobConf(chdir=True, name="default")))  # store `hydra.job.chdir=True`
    store(task)  # create & store config for our task fn
    store.add_to_hydra_store()

    zen(task).hydra_main(config_path=None, config_name="task", version_base="1.2")