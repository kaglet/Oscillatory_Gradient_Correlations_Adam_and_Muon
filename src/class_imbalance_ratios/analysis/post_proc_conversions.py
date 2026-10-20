
from src.base.lit_module.components.metrics_standard import MyMultiClassAccuracy, MyLoss
from src.base.lit_module.components.metric_gch import GCH_Metric, RGCH_Metric
from src.class_imbalance_ratios.analysis.summarise_metrics import MetricMetadata
from src.class_imbalance_ratios.analysis.needed_paths import cls_imb_outputs_dir

import os 
from pathlib import Path
import pandas as pd
import numpy as np
import torch

def delete_all_pt_extension(outputs_dir: Path) -> None:
    pt_file_paths = sorted(outputs_dir.rglob("*.pt"))
    for pt_path in pt_file_paths: 
        if ("pt" in pt_path.suffix) and (os.path.isfile(pt_path)): # do not try .is_file() of Pathlib
            os.remove(pt_path)

def convert_pt_class_counts_to_parquet(path: Path, save_path: Path) -> None: 
    class_counts = torch.load(path)
    assert type(class_counts) == dict, f"Class counts is not type dict. Got {type(class_counts)} instead."
    class_counts = {str(k) : [v] for k,v in class_counts.items()} # dict must be in form [str, array], int as key, scalar as value are not accepted
    data_df = pd.DataFrame(class_counts)
    data_df.to_parquet(save_path)      

def delete_all_non_extension(outputs_dir: Path) -> None:
    pt_file_paths = sorted(outputs_dir.rglob("*"))
    for pt_path in pt_file_paths: 
        if (not pt_path.suffix) and (os.path.isfile(pt_path)): # do not try .is_file() of Pathlib
            print(pt_path)
            os.remove(pt_path)

def convert_all_pt_metrics_to_parquet(outputs_dir: Path):
    pt_file_paths = sorted(outputs_dir.rglob("*.pt"))
    existing_parquets = sorted(outputs_dir.rglob("*.parquet"))
    summary_extra_parquets = sorted(outputs_dir.rglob("runs_extra*.parquet"))

    for pt_path in pt_file_paths:
        # Check direct conversions, not of derived metrics
        num_converted = len(existing_parquets) - len(summary_extra_parquets)
        if len(pt_file_paths) == num_converted:
            print(f"Lengths of original pt to parquet files are equal: {len(pt_file_paths)} == {num_converted}.\n Aborting conversion...")
            break
        else:
            print(f"Lengths of original pt to parquet files are unequal: {len(pt_file_paths)} > {num_converted}.\n Proceeding with conversion...")

        if Path(pt_path.parent / Path(pt_path.stem + ".parquet")).exists():
            print("File already exists at: {}".format(pt_path))
            continue
        else: 
            print("Matching parquet file does not exist at: {}. Converting...".format(pt_path))

        save_path = pt_path.parent / Path(pt_path.stem + ".parquet")
        metric = torch.load(pt_path.resolve(), map_location="cpu", weights_only=True, mmap=True)

        data_cols = [k for k in list(metric.keys()) if k != "epochs" and k != "steps"] # Do not add epochs, steps back into classes, they get merged into expected keys, and their state is reloaded when seen as part of labels, as key names, then again as metadata frmo load state dict method
        if "class_counts" in str(pt_path):
            convert_pt_class_counts_to_parquet(pt_path, save_path)
            continue # Continue after handling this file caught in regex

        # Handle metrics discovered
        if "acc" in str(pt_path):
            base_metric = MyMultiClassAccuracy(data_cols) 
        elif "loss" in str(pt_path):
            base_metric = MyLoss()
        elif "rgc" in str(pt_path).lower():
            base_metric = RGCH_Metric(data_cols)
        else:
            base_metric = GCH_Metric(data_cols)

        base_metric.load_state_dict(metric)

        computed = base_metric.compute()

        data_df = pd.DataFrame(computed).replace(float('nan'), np.nan)
        data_df = data_df.replace(100, np.nan)
        data_df = data_df.map(lambda value: value.item() if isinstance(value, torch.Tensor) else value)

        # Convert tensors not addressed by `.compute()`
        clean_epochs = np.asarray(base_metric.metric_state["epochs"]) 
        clean_steps = np.asarray(base_metric.metric_state["steps"]) 

        metadata = MetricMetadata(epochs=clean_epochs, steps=clean_steps) # Unify values in new dict
        multiindex = pd.MultiIndex.from_arrays(
            [
                list(metadata["epochs"]), 
                list(metadata["steps"]) # list of lists format required
            ],
            names=["epochs", "steps"]
        )
        data_df.index = multiindex

        data_df.to_parquet(save_path)   # by default pandas does not append .parquet      

if __name__ == "__main__":
    convert_all_pt_metrics_to_parquet(outputs_dir=cls_imb_outputs_dir)
    # delete_all_pt_extension(outputs_dir=outputs_dir)