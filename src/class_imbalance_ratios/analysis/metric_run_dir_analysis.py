""" 
Main entry point for acc analysis. 
"""
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Tuple
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd

from src.base.lit_module.components.metric_gch import GCH_Metric, RGCH_Metric
from src.base.lit_module.components.metrics_standard import MyLoss, MyMultiClassAccuracy
from src.class_imbalance_ratios.analysis.plot_managers import LonePlotOptions, LonePlotManager, show_plt_with_cleanup
from src.class_imbalance_ratios.analysis.summarise_metrics import summarize_runs_metric

def subgroup_save_acc(outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str, limit: int | None = 1) -> None:
    """ 
    Pass in list of patterns. and aggregation options, which are looped over for optional number of times. 
    """
    limit = -1 if limit is None else limit 

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
                display(f"Limit reached floor of {limit}, analysis complete")


def agg_save_metrics(outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str, limit: int | None = 1) -> None:
    """ 
    Pass in list of patterns. and aggregation options, which are looped over for optional number of times. 
    """
    limit = -1 if limit is None else limit 

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
                display(f"Limit reached floor of {limit}, analysis complete")
                break

def viz_agg_metrics(outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str) -> None:
    """ 
    Pass in list of patterns. and aggregation options, which are looped over for optional number of times. 
    """
    for experim_dir in sorted(outputs_dir.iterdir()):
        version_dirs = sorted(experim_dir.glob(version_dir))
        figax_store: list = []
        
        for i, v_dir in enumerate(version_dirs):
            # Better for figax_store to have a fixed length based of versions
            # Then index into that when repeating diff experiments
            start_overwrite_figaxs = len(figax_store) == len(version_dirs)
            if not start_overwrite_figaxs:
                figax_store.append(None)

            for info in run_paths_info:
                prefix_down = ""
                path = sorted(v_dir.rglob(info["relative_path"], case_sensitive=False))[0]
                if not Path(path).exists():
                    raise LookupError(f"Path to requested metric: {path} does not exit.")
                
                last_underscore = str(path.stem).rfind("_")
                # While still having access to v_dir with the relative path...
                # Get {Train | Validation}-{Accuracy | Loss} OR {Norm | Unnorm}-GC
                prefix_down = "Muon " if "muon" in str(experim_dir).lower() else "Adam "
                prefix_down = prefix_down + str(v_dir.stem).lower() + str(path.stem).lower()[last_underscore: ] 
                prefix_down = prefix_down.replace("-", " ")

                options = LonePlotOptions(x_label="Epochs", y_label="Accuracy", unique_label=prefix_down)
                assert type(options) is LonePlotOptions
                metric_df = pd.read_parquet(path)
                metric_df.rename(columns={"Filler":prefix_down}, inplace=True)
                plots_manager = LonePlotManager(metric_df=metric_df, plt_options=options)

                if not start_overwrite_figaxs:
                    plots_manager.viz_cols_in_one()
                    figax_store[i] = plots_manager.lone_figax # For runs in info they transfer on the same figax
                else:
                    figax = figax_store[i]
                    assert type(figax) is Tuple[Figure, Axes], f"Type of figax store at curr version index {type(figax_store[-1])} is not Tuple[Figure, Axes]"
                    plots_manager.overwrite_figax(figax=figax)
                    plots_manager.viz_cols_in_one()
    
    show_plt_with_cleanup()


if __name__ == "__main__":

    pass