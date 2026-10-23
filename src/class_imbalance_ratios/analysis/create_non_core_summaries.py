""" 
Main entry point for acc analysis. 
"""
from pathlib import Path
from typing import Any
from src.class_imbalance_ratios.analysis.summarise_metrics import summarize_runs_metric
from src.class_imbalance_ratios.analysis.needed_paths import cls_imb_s_outputs_dir, cls_imb_l_outputs_dir
from src.class_imbalance_ratios.analysis.needed_names import corr_store_names

def agg_save_metrics(outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str, limit: int | None = 1) -> None:
    """ 
    Pass in list of patterns. and aggregation options, which are looped over for optional number of times. 
    """
    limit = -1 if limit is None else limit # By pass limit with -1 as hardstop of 0 condition is never met

    for experim_dir in sorted(outputs_dir.iterdir()):
        version_dirs = sorted(experim_dir.glob(version_dir))

        for v_dir in version_dirs:
            for info in run_paths_info:
            # Get all csv file path matches to custom name for both cs metrics
                paths = sorted(v_dir.rglob(info["paths"], case_sensitive=False))
                if len(paths) > 0:
                    summarize_runs_metric(summary_mode_extras=info["summary_mode_extras"], run_metrics_paths=paths, prefix=info["prefix"])
                else:
                    raise LookupError(f"Length of paths found is {len(paths)}==0, which is not valid.")
            
            limit -= 1
            if limit == 0:
                print(f"Limit reached floor of {limit}, analysis complete")
                break

def summarize_all(dir: Path):
    """
    Generate non-core metrics from .parquet core files. 
    This operates over run directories to form different summary metrics. 
    """

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
        outputs_dir=dir, 
        version_dir="multijob/R*", 
        limit=None,

        run_paths_info=relative_paths_info
    )

    print("=========== STORING SUMMARY ACCURACY METRICS ===========") 
    relative_paths_info = [
        dict(
            paths= f"run*/train_acc.parquet", 
            prefix="Train-Accuracy", 
            summary_mode_extras={ # TODO: It is assumed the top level is ungrouped this makes if else easier, to check 1 of 2 groups first
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
        ) 
    ]

    agg_save_metrics(
        outputs_dir=dir, 
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
        outputs_dir=dir, 
        version_dir="multijob/R*", 
        limit=None,
        run_paths_info=relative_paths_info
    )

if __name__ == "__main__":
    summarize_all(cls_imb_l_outputs_dir)