import os 
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl

from src.base.lit_module.components.metrics_standard import MyMultiClassAccuracy, MyLoss
from src.base.lit_module.components.metric_gch import GCH_Metric, RGCH_Metric
from src.class_imbalance_ratios.analysis.summarise_metrics import MetricMetadata

from src.class_imbalance_ratios.analysis.post_proc_conversions import delete_all_non_extension, delete_all_pt_extension
from src.class_imbalance_ratios.analysis.post_proc_conversions import convert_all_pt_metrics_to_parquet, convert_pt_class_counts_to_parquet

from src.class_imbalance_ratios.analysis.metric_run_dir_analysis import agg_save_metrics, subgroup_save_acc
from src.class_imbalance_ratios.analysis.needed_paths import cls_imb_outputs_dir
from src.class_imbalance_ratios.analysis.needed_names import corr_store_names

from src.class_imbalance_ratios.analysis.viz_metrics import viz_lone_metrics
from src.class_imbalance_ratios.analysis.viz_metrics import viz_grid_metrics

import logging

log_file = Path(os.getcwd()) / "src/log_analysis.log"
logging.basicConfig(filename=log_file, level=logging.DEBUG)

print("=========== DELETING ALL PARQUET FILES EXPECTED BUT MISSING .PARQUET ===========")
delete_all_non_extension(outputs_dir=cls_imb_outputs_dir)

print("=========== CONVERTING ALL .PT files to .PARQUET ===========")
convert_all_pt_metrics_to_parquet(outputs_dir=cls_imb_outputs_dir) # Took 30m 50.7s on CIF10 cls_imb locally

print("=========== STORING SUMMARY GRADIENT CORRELATION METRICS ===========")
relative_paths_info = [
    dict(
        paths= f"run*/train_{corr_store_names["normed"]}.parquet", 
        prefix="Normalised-GC", 
        summary_mode_extras={
            "none": True,
            "cols": True,
            "both": True,
            "epochs": True,
            "subgroup": None
        }
    ),
    dict(
        paths = f"run*/train_{corr_store_names["raw"]}.parquet", 
        prefix="Unnormalised-GC", # For prefix which has GC majority, minority are sourced with path - groups_router (can specify in subrouter for now will hard code)
        summary_mode_extras={
            "none": True,
            "cols": True,
            "both": True,
            "epochs": True,
            "subgroup": None
        }
    )
]

agg_save_metrics(
    outputs_dir=cls_imb_outputs_dir, 
    version_dir="multijob/R*", 
    limit=None,

    run_paths_info=relative_paths_info
)

print("=========== STORING SUMMARY ACCURACY METRICS ===========") 
relative_paths_info = [
    dict(
        paths= f"run*/train_acc.parquet", 
        prefix="Train-Accuracy",
        summary_mode_extras={
            "none": True,
            "cols": True,
            "both": True,
            "epochs": True,
            "subgroup": {
                "steps": True,
                "epochs": True,
                "given_groups": {
                    "majority": [], 
                    "minority": [],
                }
            }
        }
    ),
    dict(
        paths = f"run*/val_acc.parquet", 
        prefix="Validation-Accuracy", 
        summary_mode_extras={
            "none": True,
            "cols": True,
            "both": True,
            "epochs": True,
            "subgroup": {
                "steps": True,
                "epochs": True,
                "given_groups": {
                    "majority": [], 
                    "minority": [],
                }
            }
        }
    ) # TODO: When visualizing you can superimpose similar aggregates to one plot  # TODO  We may consider decoupling atypical measures that aren't traiditional aggregation across runs e.g. subgrouping - recommended
]

agg_save_metrics(
    outputs_dir=cls_imb_outputs_dir, 
    version_dir="multijob/R*", 
    limit=None,
    run_paths_info=relative_paths_info
    
)

print("=========== STORING SUMMARY LOSS METRICS ===========")
relative_paths_info = [
    dict(
        paths= f"run*/train_loss.parquet", 
        prefix="Train-Loss",         
        summary_mode_extras={
            "none": True,
            "cols": False,
            "both": False,
            "epochs": True,
            "subgroup": None
        }
    ),     
    dict(
        paths = f"run*/val_loss.parquet", 
        prefix="Validation-Loss",         
        summary_mode_extras={
            "none": True,
            "cols": False,
            "both": False,
            "epochs": True,
            "subgroup": None
        }
    ) # TODO: When visualizing you can superimpose similar aggregates to one plot
]

agg_save_metrics(
    outputs_dir=cls_imb_outputs_dir, 
    version_dir="multijob/R*", 
    limit=None,
    run_paths_info=relative_paths_info
)

print("=========== VIZ GRAD CORR METRICS FOR REPORT ===========") 
relative_paths_info = [
    dict(
        relative_path= f"runs_extra-none_mean_Unnormalised-GC.parquet", 
    )
]

viz_grid_metrics(
    outputs_dir=cls_imb_outputs_dir, 
    version_dir="multijob/R*", 
    run_paths_info=relative_paths_info
) 

print("=========== VIZ SUMMARY ACCURACY METRICS FOR REPORT ===========") 
relative_paths_info = [
    dict(
        relative_path= f"runs_extra-both_mean_Train-Accuracy.parquet", 
    ),
    dict(
        relative_path = f"runs_extra-both_mean_Validation-Accuracy.parquet", 
    ) 
]

viz_lone_metrics(
    outputs_dir=cls_imb_outputs_dir, 
    version_dir="multijob/R*", 
    run_paths_info=relative_paths_info
) 



