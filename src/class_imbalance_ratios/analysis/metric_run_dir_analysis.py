""" 
Main entry point for acc analysis. 
"""

from pathlib import Path

from src.base.lit_module.components.metric_gch import GCH_Metric, RGCH_Metric
from src.base.lit_module.components.metrics_standard import MyLoss, MyMultiClassAccuracy
from src.class_imbalance_ratios.analysis.summarise_metrics import summarize_runs_acc, summarize_runs_gc, summarize_runs_loss

def analyse_loss(outputs_dir: Path) -> None:
    """ 
    Main entry point for loss analysis. 
    """
    for experim_dir in sorted(outputs_dir.iterdir()):
        version_dirs = sorted(experim_dir.glob("multijob/R*"))

        for v_dir in version_dirs:
            print("Train accuracy run metrics for", v_dir)

            # Get all csv file path matches to default logger file name for standard metrics
            train_loss_paths = sorted(v_dir.rglob("run*/train_loss.pt", case_sensitive=False))
            if len(train_loss_paths) > 0:
                summarize_runs_loss(run_metrics_paths= train_loss_paths, prefix=train_loss_paths[0].stem, custom_metric=MyLoss)

            print("Val accuracy run metrics for", v_dir)

            val_loss_paths = sorted(v_dir.rglob("run*/val_loss.pt", case_sensitive=False))
            if len(val_loss_paths) > 0:
                summarize_runs_loss(run_metrics_paths= val_loss_paths, prefix=val_loss_paths[0].stem, custom_metric=MyLoss)

def analyse_acc(outputs_dir: Path) -> None:
    """ 
    Main entry point for acc analysis. 
    """
    for experim_dir in sorted(outputs_dir.iterdir()):
        version_dirs = sorted(experim_dir.glob("multijob/R*"))

        for v_dir in version_dirs:
            print("Train accuracy run metrics for", v_dir)

            # Get all csv file path matches to default logger file name for standard metrics
            train_acc_paths = sorted(v_dir.rglob("run*/train_acc.pt", case_sensitive=False))
            if len(train_acc_paths) > 0:
                summarize_runs_acc(summary_mode="per_epoch", run_metrics_paths= train_acc_paths, prefix=train_acc_paths[0].stem, custom_metric=MyMultiClassAccuracy)
                summarize_runs_acc(summary_mode="global_avg", run_metrics_paths= train_acc_paths, prefix=train_acc_paths[0].stem, custom_metric=MyMultiClassAccuracy)

            print("Val accuracy run metrics for", v_dir)

            val_acc_paths = sorted(v_dir.rglob("run*/val_acc.pt", case_sensitive=False))
            if len(val_acc_paths) > 0:
                summarize_runs_acc(summary_mode="per_epoch", run_metrics_paths= val_acc_paths, prefix=val_acc_paths[0].stem, custom_metric=MyMultiClassAccuracy)
                summarize_runs_acc(summary_mode="global_avg", run_metrics_paths= val_acc_paths, prefix=val_acc_paths[0].stem, custom_metric=MyMultiClassAccuracy)

def analyse_gc(outputs_dir: Path, corr_store_names: dict[str, str] = dict(normed="GCH", raw="RGCH")) -> None:
    """ 
    Main entry point for GC analysis. 
    """
    for experim_dir in sorted(outputs_dir.iterdir()):
        version_dirs = sorted(experim_dir.glob("multijob/R*"))

        for v_dir in version_dirs:
            print("Per layer normalized gradient correlation run metrics for", v_dir)

            # Get all csv file path matches to custom name for both cs metrics
            gcs_paths = sorted(v_dir.rglob(f"run*/train_{corr_store_names["normed"]}.pt", case_sensitive=False))
            if len(gcs_paths) > 0:
                summarize_runs_gc(gcs_paths, "gcs", GCH_Metric)

            print("Per layer unnormalized gradient correlation run metrics for", v_dir)

            rgcs_paths = sorted(v_dir.rglob(f"run*/train_{corr_store_names["raw"]}.pt", case_sensitive=False))
            if len(rgcs_paths) > 0:
                summarize_runs_gc(rgcs_paths, "rgcs", RGCH_Metric)


if __name__ == "__main__":

    pass