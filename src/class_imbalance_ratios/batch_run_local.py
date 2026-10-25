import logging
import os
from pathlib import Path
import sys
# submit_grid.py
import subprocess
import os
import time
from typing import Tuple

my_log_file = Path(__file__).parent / "logs/edit_run_pbs.log"
logging.basicConfig(filename=my_log_file, level=logging.DEBUG)


def react_to_limits(msg: str, retry_counter: int, slice: int, SLEEP_TIME: int) -> dict[str, int]:
    """
    Advance loop if no limits detected, else keep loop stagnant. 
    Reset retry counter due to none busy cluster, else advance the counter to prevent infinite loop. 
    """
    if "exceed" in msg or "limit" in msg:
        time.sleep(SLEEP_TIME)
        retry_counter += 1
        slice = slice
    else:
        retry_counter = 0
        slice += 1

    return dict(retry_counter=retry_counter, slice=slice)

def react_to_fatal_err(msg: str) -> bool:
    if (not "exceed" in msg) and (not "limit" in msg):
        return True

    return False

def run_pbs_many(root_dir: Path, run_interval: list[int], slice_interval: list[int], MAX_RETRIES_IDX: int = 100):
    retry_counter = 0
    
    with open(root_dir.resolve() / "local_script.txt", "r") as f:
        template_content = f.read()

    run_length = len(run_interval)
    slice_length = len(slice_interval)
    run = 0
    while run < run_length:
        slice = 0
        while slice < slice_length:
            injected_run: int = run_interval[run]
            injected_slice: int = slice_interval[slice]

            print(f"Generating and submitting job for run={injected_run}, slice={injected_slice}")

            job_script = template_content
            job_script = job_script.replace("{{run}}", str(injected_run)).replace("{{seed}}", str(injected_run))
            job_script = job_script.replace("{{slice}}", str(injected_slice)).replace("{{ratio_seed}}", str(injected_slice))
            
            temp_filename = "temp_local_script.txt"
            with open(temp_filename, "w") as f:
                f.write(job_script)

            logging.debug("=== LOGGING PBS SCRIPT ===")
            logging.debug(job_script)

            # Launch PBS script
            attempt = subprocess.run(
                ["python", temp_filename],
                text=True, 
                shell=True, 
                capture_output=True, 
            )

            if attempt.returncode == 0:
                msg = attempt.stdout.lower()
            else:
                msg = attempt.stderr.lower()
                fatal_detected = react_to_fatal_err(msg=msg)
                if fatal_detected:
                    print(f"FATAL: Unexpected cluster error: {attempt.stderr}")
                    sys.exit(1)

            SLEEP_TIME = 60 * 60 # (sec x hr = 1hr default)
            reaction = react_to_limits(msg=msg, retry_counter=retry_counter, slice=slice, SLEEP_TIME=SLEEP_TIME)
            retry_counter = reaction["retry_counter"]
            slice = reaction["slice"]

            if retry_counter > MAX_RETRIES_IDX:
                print(f"FATAL: Reached max retry limit idx: {MAX_RETRIES_IDX}. Exiting...")
                sys.exit(1)

        run += 1

if __name__ == "__main__":
  root_dir = Path(__file__).parent.parent.parent
  print(str(root_dir))
  runs = list(range(2))
  slices = list(range(4))
  run_pbs_many(root_dir=root_dir, run_interval = runs, slice_interval = slices)