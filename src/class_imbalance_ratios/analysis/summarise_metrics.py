
"""
Accuracy metric analysis helpers. 
"""
import pandas as pd
import torch
import numpy as np
from pathlib import Path

from typing import Literal
from pandas import Index
from typing import Any, TypedDict, Type
from torchmetrics import Metric

from src.base.lit_module.components.metrics_standard import MyMultiClassAccuracy, MyLoss
from src.base.lit_module.components.metric_gch import GCH_Metric, RGCH_Metric
from src.class_imbalance_ratios.analysis.viz_metrics import viz_cols_in_one, viz_cols_in_grid

class MetricMetadata(TypedDict):
    """
    Utility unified dict for metadata stored for all the metrics, accuracy, loss, GC. 
    Useful to create a separate `pd.Dataframe`. 
    """
    epochs: np.ndarray
    steps: np.ndarray

def mean_epochs(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get 
    """
    # TODO: Find out if other index in multi-index collapses
    mean_df = data_df.groupby(level="epochs").mean(numeric_only=True) # for some reason if non-numeric added in future
    return mean_df

def mean_cols(data_df: pd.DataFrame, pooled_name: str) -> pd.DataFrame:
    mean_df = data_df.mean(axis=1).to_frame().rename(columns={0: pooled_name}) # transpose replaces lack of axes
    return mean_df

def get_next_mean_type(option: str, data_df: pd.DataFrame) -> pd.DataFrame:
    mean_df = pd.DataFrame(np.arange(0))
    if option == "epochs":
        mean_df = mean_epochs(data_df=data_df)
    elif option == "cols":
        mean_df = mean_cols(data_df=data_df, pooled_name="Filler")

    return mean_df

# TODO: agg_col_grps # If I need to subset cols 

# TODO: Do coupled Muon, Adam rather, could plot in same, for now it is fine, for all. Specify options, but may use same.
# If created agg skip over. 
def combined_mean(data_df: pd.DataFrame, options: dict[str, bool]) -> pd.DataFrame:
    mean_df = data_df
    for key, chosen in options.items():
        if chosen:
            mean_df = get_next_mean_type(option=key, data_df=mean_df)

    return mean_df

""" 
GCS analysis helpers. 
"""
def summarize_runs_metric(summary_mode_extras: Literal["epochs"] | Literal["cols"] | Literal["both"], run_metrics_paths: list[Path], prefix:str):
    """ 
    This summarises any metrics across runs. 
    Regular summary is mean and std. 
    Pandas grouping is used for extremely fast computation, no bottleneck.
    "Extras" can be attached to the summary. I.e. methods of compressing further. 
    Ensure relevance otherwise nonsensical outputs may result. 
    Output is stored in the parent directory of the current path.  

    TODO: Pass in "none" option, to have no aggregation, like before. Allow subgrouping for grouping option (see pandas book). See TODO's way above in this file. 
    """     
    run_dfs: list[pd.DataFrame] = []
    for train_metrics_path in run_metrics_paths:
        agg_df: pd.DataFrame = pd.DataFrame(np.arange(1)) # aggregated version
        data_df = pd.read_parquet(train_metrics_path)
        data_df.columns.name = "classes"
            # Across epoch aggregate - Get mean average per class 
        if summary_mode_extras == "epochs":
            agg_df = mean_epochs(data_df=data_df)  
        elif summary_mode_extras == "cols" :
            agg_df = mean_cols(data_df=data_df, pooled_name=f"Original Columns Avg - {prefix}")
        elif summary_mode_extras == "both": 
            agg_df = combined_mean(data_df=data_df, options={"epochs": True, "cols": True})
        # TODO: Add "none" option
        run_dfs.append(agg_df)

    df_all = pd.concat(run_dfs)
    runs_mean_df = df_all.groupby(level=0).mean()
    runs_std_df = df_all.groupby(level=0).std()

    parent_dir = str(run_metrics_paths[0].parent.parent.resolve())
    runs_mean_df.to_parquet(f"{parent_dir}/runs_extras::{summary_mode_extras}_mean_{prefix}")
    runs_std_df.to_parquet(f"{parent_dir}/runs_extras::{summary_mode_extras}_std_{prefix}")

if __name__ == "__main__":
    pass