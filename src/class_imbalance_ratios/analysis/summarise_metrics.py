
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
from src.class_imbalance_ratios.analysis.viz_metrics import viz_all_col_sized, viz_all_col_unsized, viz_gc_in_grid, viz_each_col_unsized, viz_loss_unsized

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

def mean_classes(data_df: pd.DataFrame) -> pd.DataFrame:
    return mean_cols(data_df=data_df, pooled_name="pooled_classes")

def mean_layers(data_df: pd.DataFrame) -> pd.DataFrame:
    return mean_cols(data_df=data_df, pooled_name="pooled_layers")

def summarize_runs_acc(summary_mode: Literal["per_epoch"] | Literal["global_avg"], run_metrics_paths: list[Path], prefix:str, custom_metric: Type[MyMultiClassAccuracy]):
    """ 
    This summarises accuracy metrics across a run.
    TODO: Output is in the parent directory of the current path.  
    """
    enumerated_metrics = enumerate(run_metrics_paths)

    sum_df: pd.DataFrame = pd.DataFrame(np.arange(1))
    metadata: MetricMetadata = MetricMetadata(epochs=np.arange(1), steps=np.arange(1))
    metadata_df: pd.DataFrame = pd.DataFrame(np.arange(1))
    sum_sq_df: pd.DataFrame = pd.DataFrame(np.arange(1))
    idx: int = 0
    agg_df: pd.DataFrame = pd.DataFrame(np.arange(1))
    run_dfs: list[pd.DataFrame] = []
    for idx, train_metrics_path in enumerated_metrics:
        metric = torch.load(train_metrics_path.resolve(), map_location="cpu")

        data_cols = [k for k in list(metric.keys()) if k != "epochs" and k != "steps"] # Do not add epochs, steps back into classes, they get merged into expected keys, and their state is reloaded when seen as part of labels, as key names, then again as metadata frmo load state dict method

        base_metric = custom_metric(data_cols) 
        base_metric.load_state_dict(metric)

        computed = base_metric.compute()

        data_df = pd.DataFrame(computed)
        data_df.replace(100, np.nan, inplace=True)
        data_df.replace(float('nan'), np.nan, inplace=True)
        data_df = data_df.map(lambda val: val.item() if isinstance(val, torch.Tensor) else val)
        
        if idx == 0:
            path_info = f"Showing run summary results in {train_metrics_path.resolve()}"
            
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
            data_df.columns.name = "classes"
            # Across epoch aggregate - Get mean average per class 
            if summary_mode == "per_epoch":
                agg_df = mean_epochs(data_df=data_df)  
            elif summary_mode == "global_avg" :
                agg_df = mean_cols(data_df=data_df, pooled_name="Class Avg")
            else: 
                raise ValueError(f"{summary_mode} is not one of options")

            run_dfs.append(agg_df)

    df_all = pd.concat(run_dfs)
    runs_mean_df = df_all.groupby(level=0).mean()
    runs_std_df = df_all.groupby(level=0).std()
    viz_loss_unsized(
        metadata_df=metadata_df,
        metric_df=runs_mean_df, 
        path_down_prefix=f"{prefix} mean"
    )    
    viz_loss_unsized(
        metadata_df=metadata_df,
        metric_df=runs_std_df, 
        path_down_prefix=f"{prefix} standard deviation"
    )

""" 
Loss analysis helpers. 
"""
def summarize_runs_loss(run_metrics_paths: list[Path], prefix:str, custom_metric: Type[MyLoss]):
    """ 
    This summarises GC metrics across a run.
    Output is in the parent directory of the current path.  
    """
    enumerated_metrics = enumerate(run_metrics_paths)

    sum_df: pd.DataFrame = pd.DataFrame(np.arange(1))
    metadata: MetricMetadata = MetricMetadata(epochs=np.arange(1), steps=np.arange(1))
    metadata_df: pd.DataFrame = pd.DataFrame(np.arange(1))
    sum_sq_df: pd.DataFrame = pd.DataFrame(np.arange(1))
    idx: int = 0
    
    for idx, train_metrics_path in enumerated_metrics:
        metric = torch.load(train_metrics_path.resolve(), map_location="cpu")

        base_metric = custom_metric() 
        base_metric.load_state_dict(metric)

        computed = base_metric.compute()

        data_df = pd.DataFrame(computed).map(lambda val: val.item() if isinstance(val, torch.Tensor) else val)
        
        if idx == 0:
            path_info = f"Showing run summary results in {train_metrics_path.resolve()}"
            
            # Convert tensors not addressed by `.compute()`
            clean_epochs = np.asarray(base_metric.metric_state["epochs"]) 
            clean_steps = np.asarray(base_metric.metric_state["steps"]) 

            metadata = MetricMetadata(epochs=clean_epochs, steps=clean_steps) # Unify values in new dict
            metadata_df = pd.DataFrame(metadata)
            
            sum_df = pd.DataFrame(data=np.zeros(data_df.shape), index=data_df.index, columns=data_df.columns)
            sum_sq_df = pd.DataFrame(data=np.zeros(data_df.shape), index=data_df.index, columns=data_df.columns)
        
        sum_df = sum_df + data_df 
        sum_sq_df = sum_sq_df + (data_df ** 2) 

    num_metrics = idx + 1 
    global_mean_df = sum_df / (idx + 1) 
    global_std_df = (sum_sq_df / num_metrics) - (global_mean_df ** 2) 

    viz_loss_unsized(
        metadata_df=metadata_df,
        metric_df=global_mean_df, 
        path_down_prefix=f"{prefix} mean"
    )    
    viz_loss_unsized(
        metadata_df=metadata_df,
        metric_df=global_std_df, 
        path_down_prefix=f"{prefix} standard deviation"
    )

""" 
GCS analysis helpers. 
"""
def summarize_runs_gc(run_metrics_paths: list[Path], prefix:str, custom_metric: Type[GCH_Metric] | Type[RGCH_Metric]):
    """ 
    This summarises GC metrics across a run.
    Output is in the parent directory of the current path.  
    """
    enumerated_metrics = enumerate(run_metrics_paths)

    sum_df: pd.DataFrame = pd.DataFrame(np.arange(1))
    metadata: MetricMetadata = MetricMetadata(epochs=np.arange(1), steps=np.arange(1))
    metadata_df: pd.DataFrame = pd.DataFrame(np.arange(1))
    sum_sq_df: pd.DataFrame = pd.DataFrame(np.arange(1))
    idx: int = 0
    
    for idx, train_metrics_path in enumerated_metrics:
        metric = torch.load(train_metrics_path.resolve(), map_location="cpu")

        data_cols = [k for k in list(metric.keys()) if k != "epochs" and k != "steps"] # Do not add epochs, steps back into classes, they get merged into expected keys, and their state is reloaded when seen as part of labels, as key names, then again as metadata frmo load state dict method

        base_metric = custom_metric(data_cols) 
        base_metric.load_state_dict(metric)

        computed = base_metric.compute()

        data_df = pd.DataFrame(computed)
        data_df.replace(100, np.nan, inplace=True)
        data_df.replace(float('nan'), np.nan, inplace=True)
        data_df = data_df.map(lambda val: val.item() if isinstance(val, torch.Tensor) else val)
        
        if idx == 0:
            path_info = f"Showing run summary results in {train_metrics_path.resolve()}"
            
            # Convert tensors not addressed by `.compute()`
            clean_epochs = np.asarray(base_metric.metric_state["epochs"]) 
            clean_steps = np.asarray(base_metric.metric_state["steps"]) 

            metadata = MetricMetadata(epochs=clean_epochs, steps=clean_steps) # Unify values in new dict
            metadata_df = pd.DataFrame(metadata)
            sum_df = pd.DataFrame(data=np.zeros(data_df.shape), index=data_df.index, columns=data_df.columns)
            sum_sq_df = pd.DataFrame(data=np.zeros(data_df.shape), index=data_df.index, columns=data_df.columns)
        
        sum_df = sum_df + data_df 
        sum_sq_df = sum_sq_df + (data_df ** 2) 

    num_metrics = idx + 1 
    global_mean_df = sum_df / (idx + 1) 
    global_std_df = (sum_sq_df / num_metrics) - (global_mean_df ** 2) 

    parent_dir = str(run_metrics_paths[0].parent.resolve())
    global_mean_df.to_parquet(f"{parent_dir}/global_mean_GC")
    global_std_df.to_parquet(f"{parent_dir}/global_std_GC")

    viz_gc_in_grid(
        metadata_df=metadata_df,
        metric_df=global_mean_df, 
        path_down_prefix=f"{prefix} mean"
    )
    viz_gc_in_grid(
        metadata_df=metadata_df,
        metric_df=global_std_df, 
        path_down_prefix=f"{prefix} standard deviation"
    )

if __name__ == "__main__":
    pass