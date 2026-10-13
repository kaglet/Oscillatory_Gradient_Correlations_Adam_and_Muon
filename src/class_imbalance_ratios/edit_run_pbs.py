import logging
import os
from pathlib import Path
import sys
# submit_grid.py
import subprocess
import os

my_log_file = Path(__file__).parent / "logs/edit_run_pbs.log"
logging.basicConfig(filename=my_log_file, level=logging.DEBUG)

def run_pbs_many(root_dir: Path, run_interval: list[int], slice_interval: list[int]):
    with open(root_dir.resolve() / "run.pbs", "r") as f:
        template_content = f.read()

    for run in run_interval:
        for slice in slice_interval:
            print(f"Generating and submitting job for run={run}, slice={slice}")

            job_script = template_content.replace("{{run}}", str(run)).replace("{{seed}}", str(run)).replace("{{slice}}", str(slice))

            temp_filename = "temp_generated_run.pbs"
            with open(temp_filename, "w") as f:
                f.write(job_script)

            logging.debug("=== LOGGING PBS SCRIPT ===")
            logging.debug(job_script)

            # Launch the PBS script via cluster commands
            subprocess.run(["qsub", temp_filename])

if __name__ == "__main__":
  root_dir = Path(__file__).parent.parent.parent
  print(str(root_dir))
  run_pbs_many(root_dir=root_dir, run_interval = [2], slice_interval = [2, 3])



