
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

def summarize_runs_metric(summary_mode_extras: list[Literal["epochs"] | Literal["cols"] | Literal["both"] | Literal["none"]], run_metrics_paths: list[Path], prefix:str):
    """ 
    This summarises any metrics across runs. 
    Regular summary is mean and std. 
    Pandas grouping is used for extremely fast computation, no bottleneck.
    "Extras" can be attached to the summary. I.e. methods of compressing further. 
    Ensure relevance otherwise nonsensical outputs may result. 
    Output is stored in the parent directory of the current path.  

    TODO: Allow subgrouping for grouping separately (do NOT convolute this streamlined logic with highly custom logic). See TODO's way above in this file. 
    """     
    per_aggregator_store: list[list[Any]] = [list() for _ in summary_mode_extras]
    for train_metrics_path in run_metrics_paths:
        agg_df: pd.DataFrame = pd.DataFrame(np.arange(1)) # aggregated version
        data_df = pd.read_parquet(train_metrics_path)
        data_df.columns.name = "classes"
        
        for extra in summary_mode_extras:
            if extra == "epochs":
                agg_df = mean_epochs(data_df=data_df)  
                per_aggregator_store[0].append(agg_df)
            elif extra == "cols" :
                agg_df = mean_cols(data_df=data_df, pooled_name=f"Original Columns Avg - {prefix}")
                per_aggregator_store[1].append(agg_df)
            elif extra == "both": 
                agg_df = combined_mean(data_df=data_df, options={"epochs": True, "cols": True})
                per_aggregator_store[2].append(agg_df)
            elif extra == "none": 
                agg_df = data_df # no change
                per_aggregator_store[3].append(agg_df)

        # TODO: Test code will reach when it gets here
        for i, dfs_list in enumerate(per_aggregator_store):
            df_all = pd.concat(dfs_list)
            assert type(df_all) is pd.DataFrame, f"List of type {type(df_all)} could not be aggregated into pd.Dataframe"
            runs_mean_df = df_all.groupby(level=0).mean()
            runs_std_df = df_all.groupby(level=0).std()

            parent_dir = str(run_metrics_paths[0].parent.parent.resolve())
            runs_mean_df.to_parquet(f"{parent_dir}/runs_extras::{summary_mode_extras[i]}_mean_{prefix}") # Decide how to display later, do not plot separately yet for now, generic plot method will have more flexibility
            runs_std_df.to_parquet(f"{parent_dir}/runs_extras::{summary_mode_extras[i]}_std_{prefix}")

if __name__ == "__main__":
    pass