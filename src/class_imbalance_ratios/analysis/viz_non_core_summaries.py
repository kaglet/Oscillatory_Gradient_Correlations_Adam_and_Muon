from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
from src.class_imbalance_ratios.analysis.needed_paths import cls_imb_outputs_dir
from src.class_imbalance_ratios.analysis.plot_managers import GridPlotOptions, GridPlotsmanager, LonePlotOptions, LonePlotManager

def viz_lone_metrics(outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str) -> None:
    """ 
    Pass in list of patterns. and aggregation options, which are looped over for optional number of times. 
    """
    figax_store: list = []
    plots_managers = []
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
                
                first_hyphen = str(path.stem).find("-")
                # While still having access to v_dir with the relative path...
                # Get {Train | Validation}-{Accuracy | Loss} OR {Norm | Unnorm}-GC
                prefix_down = "Muon " if "muon" in str(experim_dir).lower() else "Adam "
                prefix_down = prefix_down + str(v_dir.stem).lower() + str(path.stem).lower()[first_hyphen: ] 
                prefix_down = prefix_down.replace("-", " ").replace("_", " ")

                x_label = "Epochs" if "epochs" in path.stem.lower() else "Steps"

                options = LonePlotOptions(x_label=x_label, y_label="Accuracy", unique_label=prefix_down)
                assert type(options) is LonePlotOptions
                metric_df = pd.read_parquet(path)
                metric_df.rename(columns={"Filler":prefix_down}, inplace=True)

                plots_manager = LonePlotManager(metric_df=metric_df, plt_options=options)

                if not start_overwrite_figaxs and figax_store[i] is None:
                    figax_store[i] = plots_manager # For runs in info they transfer on the same figax
                else:
                    used_manager = figax_store[i]
                    assert type(used_manager) is LonePlotManager, ""
                    plots_manager.overwrite_figax(figax=used_manager.lone_figax)

                plots_managers.append(plots_manager)

    for mngr in plots_managers: 
        assert type(mngr) is LonePlotManager, ""
        mngr.viz_cols_in_one()
    
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
                first_hyphen = str(path.stem).find("-")
                # While still having access to v_dir with the relative path...
                # Get {Train | Validation}-{Accuracy | Loss} OR {Norm | Unnorm}-GC
                prefix_down = "Muon " if "muon" in str(experim_dir).lower() else "Adam "
                if "subgroup_steps" in str(path.stem).lower():
                    prefix_down = prefix_down + "subgroup_steps "
                elif "subgroup_epochs" in str(path.stem).lower():
                    prefix_down = prefix_down + "subgroup_epochs "

                prefix_down = prefix_down + str(v_dir.stem).lower() + str(path.stem).lower()[first_hyphen: ] 
                prefix_down = prefix_down.replace("-", " ").replace("_", " ")

                metric_df = pd.read_parquet(path)
                x_label = "Epochs" if "epochs" in path.stem.lower() else "Steps"
                # metric_df.rename(columns={"Filler":prefix_down}, inplace=True)
                options = GridPlotOptions(x_label=x_label, y_label="Dot Product", grid_title=prefix_down, grid_cols_num=5)
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
    """ 
    Visualize non-core metrics that were created. 
    """
    # TODO: Do this again but for STD not mean (replace or do loop for modes (easiest))
    print("=========== VIZ UNNORMED GRAD CORR METRICS FOR REPORT ===========") 
    # Unnormed Per col per step (per OPTIM step)
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

    # Unnormed Col reduction per step (per OPTIM step)
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-cols_mean_Unnormalised-GC.parquet", 
        )
    ]

    viz_lone_metrics(
        outputs_dir=cls_imb_outputs_dir, 
        version_dir="multijob/R*", 
        run_paths_info=relative_paths_info
    ) 

    print("=========== VIZ NORMED GRAD CORR METRICS FOR REPORT ===========") 
    # Normed Per col per step (per OPTIM step)
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-none_mean_Normalised-GC.parquet", 
        )
    ]

    viz_grid_metrics(
        outputs_dir=cls_imb_outputs_dir, 
        version_dir="multijob/R*", 
        run_paths_info=relative_paths_info
    ) 

    # Normed Col reduction per step (per OPTIM step)
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-cols_mean_Normalised-GC.parquet", 
        )
    ]

    viz_lone_metrics(
        outputs_dir=cls_imb_outputs_dir, 
        version_dir="multijob/R*", 
        run_paths_info=relative_paths_info
    ) 

    print("=========== VIZ SUMMARY ACCURACY METRICS FOR REPORT ===========") 
    # Cols present per epoch
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-none_mean_Train-Accuracy.parquet", 
        ),
        dict(
            relative_path = f"runs_extra-none_mean_Validation-Accuracy.parquet", 
        ) 
    ]

    viz_lone_metrics(
        outputs_dir=cls_imb_outputs_dir, 
        version_dir="multijob/R*", 
        run_paths_info=relative_paths_info
    ) 
    
    # Col reduction per epoch
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

    # Subgroups per epoch
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-subgroup_epochs_mean_Train-Accuracy.parquet", 
        ),
        dict(
            relative_path = f"runs_extra-subgroup_epochs_mean_Validation-Accuracy.parquet", 
        ) 
    ]

    viz_lone_metrics(
        outputs_dir=cls_imb_outputs_dir, 
        version_dir="multijob/R*", 
        run_paths_info=relative_paths_info
    ) 

    print("=========== VIZ SUMMARY LOSS METRICS FOR REPORT ===========") 
    # Loss per step (NOT per optim step)
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-none_mean_Train-Loss.parquet", 
        ),
        dict(
            relative_path = f"runs_extra-none_mean_Validation-Loss.parquet", 
        ) 
    ]

    viz_lone_metrics(
        outputs_dir=cls_imb_outputs_dir, 
        version_dir="multijob/R*", 
        run_paths_info=relative_paths_info
    ) 


