import logging
import os
from pathlib import Path
import sys

if __name__ == "__main__":
  root_dir = Path("/mnt/lustre/users/kmotlhabane/Oscillatory_Gradient_Correlations_Adam_and_Muon/")
  trainer_dir = Path(root_dir/ "")
  runtime_experim_dir = Path(root_dir / "outputs/${hydra.job.name}/multijob/R_${counts_prepper.downsampler_ratio}")
  runtime_run_dir = "./run_${run_idx}/" # This path is relative to cwd (current working dir) set by Hydra during program run, could use an interpolated function
  log_file = Path(root_dir / "src/class_imbalance_ratios/tests.log")
      
  data_dir = Path(root_dir / "data/")  
  logging.basicConfig(filename=log_file, level=logging.DEBUG)

  logging.debug("root_dir: {} \n".format(root_dir))
  logging.debug("trainer_dir {} \n".format(trainer_dir))
  logging.debug("runtime_experim_dir {} \n".format(runtime_experim_dir))
  logging.debug("runtime_run_dir {} \n".format(runtime_run_dir))
  logging.debug("log_file {} \n".format(log_file))
  logging.debug("data_dir {} \n".format(data_dir))



