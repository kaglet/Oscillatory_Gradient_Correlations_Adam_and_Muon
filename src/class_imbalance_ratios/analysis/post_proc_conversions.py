
from src.base.lit_module.components.metrics_standard import MyMultiClassAccuracy, MyLoss
from src.base.lit_module.components.metric_gch import GCH_Metric, RGCH_Metric
from src.class_imbalance_ratios.analysis.summarise_metrics import MetricMetadata
from src.class_imbalance_ratios.analysis.needed_paths import outputs_dir

import os 
from IPython.display import display
from pathlib import Path
import pandas as pd
import numpy as np
import torch

def convert_pt_class_counts_to_parquet(path: Path) -> None: 
    class_counts = torch.load(path)
    assert type(class_counts) == dict, f"Class counts is not type dict. Got {type(class_counts)} instead."
    class_counts = {str(k) : [v] for k,v in class_counts.items()} # dict must be in form [str, array], int as key, scalar as value are not accepted
    data_df = pd.DataFrame(class_counts)
    data_df.to_parquet(path.parent / path.stem)      

def delete_all_non_extension(outputs_dir: Path) -> None:
    pt_file_paths = sorted(outputs_dir.rglob("*"))
    for pt_path in pt_file_paths: 
        if (not pt_path.suffix) and (os.path.isfile(pt_path)): # do not try .is_file() of Pathlib
            display(pt_path)
            os.remove(pt_path)

def delete_all_pt_extension(outputs_dir: Path) -> None:
    pt_file_paths = sorted(outputs_dir.rglob("*"))
    for pt_path in pt_file_paths: 
        if (pt_path.suffix == "pt") and (os.path.isfile(pt_path)): # do not try .is_file() of Pathlib
            os.remove(pt_path)

def convert_all_pt_metrics_to_parquet(outputs_dir: Path):
    pt_file_paths = sorted(outputs_dir.rglob("*.pt"))
    existing_parquets = sorted(outputs_dir.rglob("*.parquet*"))
    if len(pt_file_paths) == len(existing_parquets):
        display(f"Lengths of pt to parquet files are equal: {len(pt_file_paths)} == {len(existing_parquets)}.\n Aborting conversion...")
    for pt_path in pt_file_paths:
        metric = torch.load(pt_path.resolve(), map_location="cpu", weights_only=True, mmap=True)

        data_cols = [k for k in list(metric.keys()) if k != "epochs" and k != "steps"] # Do not add epochs, steps back into classes, they get merged into expected keys, and their state is reloaded when seen as part of labels, as key names, then again as metadata frmo load state dict method
        if "class_counts" in str(pt_path):
            convert_pt_class_counts_to_parquet(pt_path)
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

        data_df.to_parquet(pt_path.parent / Path(pt_path.stem + ".parquet") )   # by default pandas does not append .parquet   

if __name__ == "__main__":
    convert_all_pt_metrics_to_parquet(outputs_dir=outputs_dir)
    # delete_all_pt_extension(outputs_dir=outputs_dir)