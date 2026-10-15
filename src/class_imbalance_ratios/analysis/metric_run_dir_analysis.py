""" 
Main entry point for acc analysis. 
"""

from pathlib import Path
from IPython.display import display

from src.base.lit_module.components.metric_gch import GCH_Metric, RGCH_Metric
from src.base.lit_module.components.metrics_standard import MyLoss, MyMultiClassAccuracy
from src.class_imbalance_ratios.analysis.summarise_metrics import summarize_runs_metric

def analyse_loss(outputs_dir: Path, limit: int | None = 1) -> None:
    """ 
    Main entry point for loss analysis. 
    """
    limit = 0 if limit is None else limit 

    for experim_dir in sorted(outputs_dir.iterdir()):
        # TODO: This format cannot be flattened. Specify 2 patterns: for experiment in multijob e.g. R as the "version" or "slice" even, and another for across the runs inside that, to only aggreg there.
        version_dirs = sorted(experim_dir.glob("multijob/R*"))

        for v_dir in version_dirs:

            print("Train accuracy run metrics for", v_dir)

            # Get all csv file path matches to default logger file name for standard metrics
            train_loss_paths = sorted(v_dir.rglob("run*/train_loss.parquet", case_sensitive=False))
            if len(train_loss_paths) > 0:
                summarize_runs_metric(summary_mode_extras="cols", run_metrics_paths= train_loss_paths, prefix=train_loss_paths[0].stem)

            print("Val accuracy run metrics for", v_dir)

            val_loss_paths = sorted(v_dir.rglob("run*/val_loss.parquet", case_sensitive=False))
            if len(val_loss_paths) > 0:
                summarize_runs_metric(summary_mode_extras="cols", run_metrics_paths= val_loss_paths, prefix=val_loss_paths[0].stem)
            
            limit -= 1
            if limit == 0:
                display(f"Limit reached floor of {limit}, analysis complete")
                
def analyse_acc(outputs_dir: Path, limit: int | None = 1) -> None:
    """ 
    Main entry point for acc analysis. 
    """
    limit = 0 if limit is None else limit 
    
    for experim_dir in sorted(outputs_dir.iterdir()):
        version_dirs = sorted(experim_dir.glob("multijob/R*"))

        for v_dir in version_dirs:
            print("Train accuracy run metrics for", v_dir)

            # Get all csv file path matches to default logger file name for standard metrics
            train_acc_paths = sorted(v_dir.rglob("run*/train_acc.parquet", case_sensitive=False))
            if len(train_acc_paths) > 0:
                summarize_runs_metric(summary_mode_extras="epochs", run_metrics_paths= train_acc_paths, prefix=train_acc_paths[0].stem)
                summarize_runs_metric(summary_mode_extras="cols", run_metrics_paths= train_acc_paths, prefix=train_acc_paths[0].stem)
                summarize_runs_metric(summary_mode_extras="both", run_metrics_paths= train_acc_paths, prefix=train_acc_paths[0].stem)

            print("Val accuracy run metrics for", v_dir)

            val_acc_paths = sorted(v_dir.rglob("run*/val_acc.parquet", case_sensitive=False))
            if len(val_acc_paths) > 0:
                summarize_runs_metric(summary_mode_extras="epochs", run_metrics_paths= val_acc_paths, prefix=val_acc_paths[0].stem)
                summarize_runs_metric(summary_mode_extras="cols", run_metrics_paths= val_acc_paths, prefix=val_acc_paths[0].stem)
                summarize_runs_metric(summary_mode_extras="both", run_metrics_paths= val_acc_paths, prefix=val_acc_paths[0].stem)

            limit -= 1
            if limit == 0:
                display(f"Limit reached floor of {limit}, analysis complete")

def agg_save_metrics(outputs_dir: Path, limit: int | None = 1, corr_store_names: dict[str, str] = dict(normed="GCH", raw="RGCH")) -> None:
    """ 
    Pass in paths, and aggregation options, which are looped over for optional number of times. 
    """
    limit = 0 if limit is None else limit 

    for experim_dir in sorted(outputs_dir.iterdir()):
        version_dirs = sorted(experim_dir.glob("multijob/R*"))

        for v_dir in version_dirs:
            print("Per layer normalized gradient correlation run metrics for", v_dir)

            # Get all csv file path matches to custom name for both cs metrics
            gcs_paths = sorted(v_dir.rglob(f"run*/train_{corr_store_names["normed"]}.parquet", case_sensitive=False))
            if len(gcs_paths) > 0:
                summarize_runs_metric(summary_mode_extras="epochs", run_metrics_paths=gcs_paths, prefix="gcs")

            print("Per layer unnormalized gradient correlation run metrics for", v_dir)

            rgcs_paths = sorted(v_dir.rglob(f"run*/train_{corr_store_names["raw"]}.parquet", case_sensitive=False))
            if len(rgcs_paths) > 0:
                summarize_runs_metric(summary_mode_extras="epochs", run_metrics_paths=gcs_paths, prefix="rgcs")

            limit -= 1
            if limit == 0:
                display(f"Limit reached floor of {limit}, analysis complete")

if __name__ == "__main__":

    pass