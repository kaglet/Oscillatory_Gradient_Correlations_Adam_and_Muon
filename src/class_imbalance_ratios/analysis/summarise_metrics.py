
"""
Accuracy metric analysis helpers. 
"""
import pandas as pd
import torch
import numpy as np
from pathlib import Path

from IPython.display import display

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

def mean_subgroups(data_df: pd.DataFrame, given_groups: dict) -> pd.DataFrame:
    """

    Right here change type expectation of groups to bubble up, to be key items i.e. name of group then the list as value, not just the list, needs a name, to name the dataframe result. I could create custom type but won't. I will manually type it out by keyboard. 
    """
    col_dfs = []
    for name, col_subset in given_groups.items():
        mean_df = data_df[col_subset].mean(axis=1).to_frame(name=name) # for some reason if non-numeric added in future
        col_dfs.append(mean_df)

    grouped_df = pd.concat(col_dfs, axis=1) 
    return grouped_df

def mean_cols(data_df: pd.DataFrame, pooled_name: str) -> pd.DataFrame:
    mean_df = data_df.mean(axis=1).to_frame().rename(columns={0: pooled_name}) # transpose replaces lack of axes
    return mean_df

def get_subgroup_type(option: tuple[str, Any], data_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    extra, val = option
    mean_df = pd.DataFrame(np.arange(0))

    # Copious asserts, could be amended via a typed dict structure of Literal values expected, or a class of repeatable instances.
    assert type(val) is dict, f"Value attached to subgroup must be dict. Got {type(val)} instead."
    assert type(val["epochs"]) is bool, f"Expected val of expected key 'epochs' to be boolean. Got {type(val["epochs"])} instead."

    assert type(val["given_groups"]) is dict, f"Groups given of {type(val["given_groups"])} are not of dict"
    assert len(val["given_groups"]) >= 2, f"Groups given of length {len(val["given_groups"])} must be at least 2."

    single_aggregated_store: dict[str, pd.DataFrame] = {
        f"{extra}_epochs": pd.DataFrame(np.arange(0)), 
        f"{extra}_steps": pd.DataFrame(np.arange(0))
    }
    mean_df = data_df 

    # Perform reduction then epochs if both are true
    if val["steps"] == True: 
        mean_df = mean_subgroups(data_df=mean_df, given_groups=val["given_groups"])
        single_aggregated_store[f"{extra}_steps"] = mean_df
    
    if val["epochs"] == True: 
        mean_df = mean_epochs(data_df=mean_df)
        single_aggregated_store[f"{extra}_epochs"] = mean_df

    if val["steps"] == False and val["epochs"] == False:
        raise ValueError(
            "For subgroup both `steps` and `epochs` are both set to `False`.\n" \
            "Aggregation cannot be applied. Set one or both to `False` and retry.\n" \
            "While returning unchanged dataframe is possible, this duplicates dataframe in 'none' option" \
            "twice under each new subgroup `final_name` in @summarize_runs_metric.\n" \
            "This script prevents incorrect metric types under a protected name.\n" \
            "Aborting..."
        )

    return single_aggregated_store

def get_next_mean_type(option: tuple[str, Any], data_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Details of how to parse it is left up to the method below. 
    val can be of type Any but type expectations are coupled with the extra given.
    """
    extra, val = option
    mean_df = pd.DataFrame(np.arange(0))
    if extra == "epochs" and val == True:
        mean_df = mean_epochs(data_df=data_df)
    elif extra == "cols" and val == True:
        mean_df = mean_cols(data_df=data_df, pooled_name="Filler")

    return (mean_df, extra)

# TODO: agg_col_grps # If I need to subset cols 

def dynamic_append_to_dict(some_dict: dict[str, list[Any]], key:str, val: Any) -> dict[str, Any]:
    some_dict.setdefault(key, []).append(val)

    return some_dict

# TODO: Do coupled Muon, Adam rather, could plot in same, for now it is fine, for all. Specify options, but may use same.
# If created agg skip over. 
def summarize_runs_metric(summary_mode_extras: dict[Literal["epochs"] | Literal["cols"] | Literal["both"] | Literal["none"], Any], run_metrics_paths: list[Path], prefix:str):
    """ 
    This summarises any metrics across runs. 
    Regular summary is mean and std. 
    Pandas grouping is used for extremely fast computation, no bottleneck.
    "Extras" can be attached to the summary. I.e. methods of compressing a run's metrics further, 
    aggregating them, and taking/storing only the aggregated result we are typically interested in compared to one particular run. 
    Ensure relevance otherwise nonsensical outputs may result. 
    Output is stored in the parent directory of the current path.  
    """   
    parent_dir = str(run_metrics_paths[0].parent.parent.resolve())
  
    per_aggregator_store: dict[str, list[Any]] = {extra: list() for extra in summary_mode_extras}
    for train_metrics_path in run_metrics_paths:
        agg_df: pd.DataFrame = pd.DataFrame(np.arange(1)) # aggregated version
        data_df = pd.read_parquet(train_metrics_path)
        data_df.columns.name = f"Classes"
        
        for extra, val in summary_mode_extras.items():
            path_to_extra = f"{parent_dir}/runs_extra-{extra}_mean_{prefix}.parquet"
            if Path(path_to_extra).exists():
                continue 
            else: 
                display(f"Extra does not exist for: {parent_dir}/runs_extra-{extra}_mean_{prefix}.parquet \n Creating by aggregation...")

            """
            TODO: 
            # Dynamic append to dictionary if key does not exist. Allows dynamic setting beyond typical hierarchy struct can be better enforced. 
            # Ensure knowledge of final aggregation metric e.g. name, is returned by the method forming the aggregation itself, not just the initial extra name in potential short form. 
            # Subgrouping order is maintained with another dictionary setting of configuration of its value. 
            # Converted fragile lists to dict of keys with value to be considered.
            # Knowledge of values is deferred to functions that need them, and possibly with the key the value came with, instead of the entire val dictionary - assuming that is the value. 
            """
            if extra == "epochs" and val == True:
                agg_df = mean_epochs(data_df=data_df) 
                per_aggregator_store["epochs"].append(agg_df)
            elif extra == "cols" and val == True:
                agg_df = mean_cols(data_df=data_df, pooled_name=f"Original Columns Avg - {prefix}")
                per_aggregator_store["cols"].append(agg_df)
            elif extra == "both" and val == True: 
                (agg_df, final_name) = get_next_mean_type(option=(extra, val), data_df=data_df)
                (agg_df, final_name) = get_next_mean_type(option=(extra, val), data_df=agg_df) 
                per_aggregator_store = dynamic_append_to_dict(some_dict=per_aggregator_store, key=final_name, val=agg_df)
            elif extra == "subgroup" and val is not None: 
                if "Train-Accuracy" in prefix or "Validation-Accuracy" in prefix:
                    counts_df = pd.read_parquet(train_metrics_path.parent / "train_actual_class_counts.parquet").T
                    labelled_df = pd.read_parquet(train_metrics_path.parent / "train_desired_class_counts.parquet").T
                    counts_df.index = labelled_df.index
                    counts_series = counts_df.squeeze(axis=1)
                    assert type(counts_series) is pd.Series, f"Expected counts_series of {type(counts_series)} to be pd.Series."
                    min = counts_series.min()
                    max = counts_series.max()
                    min_series = counts_series[counts_series == min] 
                    max_series = counts_series[counts_series != min]
                    min_list = [lbl for lbl, count in min_series.to_dict().items()]
                    max_list = [lbl for lbl, count in max_series.to_dict().items()]
                    val["given_groups"]["minority"] = min_list 
                    val["given_groups"]["majority"] = max_list

                    # Alter prefix to reflect extended title
                    # prefix += f"-Majority-count-{int(max)}---Minority-count-{int(min)}"

                single_aggregated_store = get_subgroup_type(option=(extra, val), data_df=data_df) # Refactor to delegate key val checking to the function and usage of the subgroups for aggregation, type checking of Any happens for preset pairs by exclusion
                for final_name, agg_df in single_aggregated_store.items(): # When deferring funcs may return deferred keys-vals to append to this aggregator store 
                    per_aggregator_store = dynamic_append_to_dict(some_dict=per_aggregator_store, key=final_name, val=agg_df)
            elif extra == "none": 
                agg_df = data_df # no change
                per_aggregator_store[f"{extra}"].append(agg_df)

        for extra, dfs_list in per_aggregator_store.items():
            if len(dfs_list) == 0:
                # display(f"Extra exists for: {parent_dir}/runs_extra-{extra}_mean_{prefix}.parquet \n Aborting aggregation and store ...")
                continue 

            df_all = pd.concat(dfs_list)
            assert type(df_all) is pd.DataFrame, f"List of type {type(df_all)} for extra {extra} could not be aggregated into pd.Dataframe"
            runs_mean_df = df_all.groupby(level=0).mean()
            runs_std_df = df_all.groupby(level=0).std()

            runs_mean_df.to_parquet(path_to_extra) # Decide how to display later, do not plot separately yet for now, generic plot method will have more flexibility
            runs_std_df.to_parquet(path_to_extra)

if __name__ == "__main__":
    pass