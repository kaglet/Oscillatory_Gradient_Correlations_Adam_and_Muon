from dataclasses import dataclass
import os
from pathlib import Path

__all__ = ["Experiment_Data"]

@dataclass
class Experiment_Data:
    name: str = "unnamed"
    config_postfix: str = "runs"
    run_num: int = 0

    def __post_init__(self):
        global_output_path = Path(f"outputs/{self.name}" )
        os.makedirs(global_output_path, exist_ok=True)
        all_outputs = sorted([local_path for local_path in os.listdir(global_output_path)])  
        output_path = ""
        if len(all_outputs) == 0:
            output_path = global_output_path / "v_0"
        else:
            last = all_outputs[-1]
            next_i = str(int(last[-1])+1)
            output_path = global_output_path / f"v_{next_i}"

        self.log_path = Path(f"{output_path}/{self.config_postfix}/" )
        os.makedirs(self.log_path, exist_ok=True)

if __name__ == "__main__":
    pass 