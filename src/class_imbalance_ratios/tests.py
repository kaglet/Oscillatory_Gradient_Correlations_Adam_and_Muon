import logging
import os
from pathlib import Path
import sys
import torch 
import torch.nn.functional as F

if __name__ == "__main__":
  # root_dir = Path("/mnt/lustre/users/kmotlhabane/Oscillatory_Gradient_Correlations_Adam_and_Muon/")

  # my_log_file = Path(root_dir / "src/class_imbalance_ratios/logs/chpc_local_dir_tests.log")
  # logging.basicConfig(filename=my_log_file, level=logging.DEBUG)
  
  # trainer_dir = Path(root_dir/ "")
  # runtime_experim_dir = Path(root_dir / "outputs/${hydra.job.name}/multijob/R_${counts_prepper.downsampler_ratio}")
  # runtime_run_subdir = "${hydra:runtime.output_dir}/run_${run_idx}/" # This path is relative to cwd (current working dir) set by Hydra during program run, could use an interpolated function
  # data_dir = Path(root_dir / "data/")  

  # logging.debug("root_dir: {} \n".format(root_dir))
  # logging.debug("trainer_dir {} \n".format(trainer_dir))
  # logging.debug("runtime_experim_dir {} \n".format(runtime_experim_dir))
  # logging.debug("runtime_run_subdir {} \n".format(runtime_run_subdir))
  # logging.debug("log_file {} \n".format(my_log_file))
  # logging.debug("data_dir {} \n".format(data_dir))

  prev_grad = torch.randn(100, 128)
  curr_grad = torch.randn(100, 128)
  my_output = torch.sum(torch.mul(prev_grad, curr_grad)) / (prev_grad.norm() * curr_grad.norm())

  with_dot = torch.dot(prev_grad.reshape(-1), curr_grad.reshape(-1)) / (prev_grad.norm() * curr_grad.norm())

  prev_grad = prev_grad.reshape(-1) 
  curr_grad = curr_grad.reshape(-1)
  output = F.cosine_similarity(prev_grad, curr_grad, dim=0)

  print("Torch cosine", output)
  print("My output", my_output)
  print("With torch dot", with_dot)
  print(torch.equal(output, my_output))



