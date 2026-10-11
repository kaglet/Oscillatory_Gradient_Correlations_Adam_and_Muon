import logging
from pathlib import Path
import sys

if __name__ == "__main__":
  root_dir = Path(__file__).parent.parent.parent
  trainer_dir = Path(root_dir)
  runtime_experim_dir = str(root_dir) + "outputs/${hydra.job.name}/multijob/R_${counts_prepper.downsampler_ratio}"
  runtime_run_dir = "./run_${run_idx}/" # This path is relative to cwd (current working dir) set by Hydra during program run
  log_file = Path(__file__).parent / "cls_imb_logs.log"

  data_dir = str(Path(root_dir / "data/").resolve())  
  logging.basicConfig(filename=log_file, level=logging.DEBUG)

  logging.debug("root_dir: {} \n".format(root_dir))
  logging.debug("trainer_dir {} \n".format(trainer_dir))
  logging.debug("runtime_experim_dir {} \n".format(runtime_experim_dir))
  logging.debug("runtime_run_dir {} \n".format(runtime_run_dir))
  logging.debug("log_file {} \n".format(log_file))
  logging.debug("data_dir {} \n".format(data_dir))



