from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
from src.class_imbalance_ratios.analysis.plot_managers import GridPlotOptions, GridPlotsmanager, LonePlotOptions, LonePlotManager

def viz_lone_metrics(outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str) -> None:
    """ 
    Pass in list of patterns. and aggregation options, which are looped over for optional number of times. 
    """
    figax_store: list = []
    for experim_dir in sorted(outputs_dir.iterdir()):
        version_dirs = sorted(experim_dir.glob(version_dir))
        
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

                if not start_overwrite_figaxs and figax_store[i] is None:
                    plots_manager.viz_cols_in_one()
                    figax_store[i] = plots_manager.lone_figax # For runs in info they transfer on the same figax
                else:
                    fig, ax = figax_store[i]
                    figax = (fig, ax)
                    assert type(fig) is Figure and type(ax) is Axes, f"Type of figax store at curr version index {type(figax_store[-1])} is not Tuple[Figure, Axes]"
                    plots_manager.overwrite_figax(figax=figax)
                    plots_manager.viz_cols_in_one()
    
def viz_grid_metrics(outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str) -> None:
    """ 
    Pass in list of patterns. and aggregation options, which are looped over for optional number of times. 
    """
    plots_managers = []
    figaxs_store: list = []
    for experim_dir in sorted(outputs_dir.iterdir()):
        version_dirs = sorted(experim_dir.glob(version_dir))
        
        for i, v_dir in enumerate(version_dirs):
            # Better for figax_store to have a fixed length based of versions
            # Then index into that when repeating diff experiments
            start_overwrite_figaxs = len(figaxs_store) == len(version_dirs)
            if not start_overwrite_figaxs:
                figaxs_store.append(None)

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

                metric_df = pd.read_parquet(path)
                # metric_df.rename(columns={"Filler":prefix_down}, inplace=True)
                options = GridPlotOptions(x_label="Epochs", y_label="Dot Product", grid_title=prefix_down, grid_cols_num=5)
                assert type(options) is GridPlotOptions


                plots_manager = GridPlotsmanager(metric_df=metric_df, plt_options=options)

                if not start_overwrite_figaxs and figaxs_store[i] is None:
                    figaxs_store[i] = plots_manager # For runs in info they transfer on the same figax
                else:
                    used_manager = figaxs_store[i]
                    assert type(used_manager) is GridPlotsmanager, ""
                    plots_manager.overwrite_figaxes(figaxes=used_manager.grid_figaxs)

                plots_managers.append(plots_manager)

    for mngr in plots_managers: 
        assert type(mngr) is GridPlotsmanager, ""
        mngr.viz_cols_in_grid()
        
if __name__ == "__main__":
    pass