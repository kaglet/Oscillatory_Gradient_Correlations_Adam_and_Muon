import os
from pathlib import Path
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
from src.class_imbalance_ratios.analysis.needed_paths import cls_imb_s_outputs_dir, cls_imb_l_outputs_dir
from src.class_imbalance_ratios.analysis.plot_managers import GridPlotOptions, GridPlotsmanager, LonePlotOptions, LonePlotManager

def viz_lone_metrics(mode:str, outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str, final_fig_name:str) -> None:
    """ 
    Pass in list of patterns. and aggregation options, which are looped over for optional number of times. 
    """
    figax_store: list = []
    plots_managers = []

    for experim_dir in sorted(outputs_dir.iterdir()):
        version_dirs = sorted(experim_dir.glob(version_dir))
        
        for i, v_dir in enumerate(version_dirs):
            print(f"v_dir: {v_dir.stem} matched to idx: {i}")
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
                
                # While still having access to v_dir with the relative path...
                # Get {Train | Validation}-{Accuracy | Loss} OR {Norm | Unnorm}-GC
                optimizer_name = "Muon " if "muon" in str(experim_dir).lower() else "Adam "
                prefix_down = optimizer_name
                if "subgroup_steps" in str(path.stem).lower():
                    prefix_down = prefix_down + "subgroup_steps "
                elif "subgroup_epochs" in str(path.stem).lower():
                    prefix_down = prefix_down + "subgroup_epochs "

                last_underscore = str(path.stem).rfind("_")
                version = str(v_dir.stem).lower()
                extra_name_full = str(path.stem).lower()[last_underscore: ] 
                prefix_down = prefix_down + extra_name_full
                prefix_down = prefix_down.replace("-", " ").replace("_", " ").capitalize()

                x_label = "Epochs" if "epochs" in path.stem.lower() else "Steps"
                y_label = "Unknown"
                if "accuracy" in path.stem.lower():
                    y_label = "Accuracy" 
                elif "loss" in path.stem.lower():
                    y_label = "Loss" 
                elif "unnormalised" in path.stem.lower():
                    y_label = "Dot Product"
                else:
                    y_label = "Cosine Similarity"

                optimizer_name = optimizer_name.capitalize()
                mode = mode.capitalize()
                final_fig_name=final_fig_name.capitalize()
                options = LonePlotOptions(base_size=8, x_label=x_label, y_label=y_label, lone_title= optimizer_name + " " + mode + final_fig_name, prefix=prefix_down)
                assert type(options) is LonePlotOptions
                metric_df = pd.read_parquet(path)
                metric_df.rename(columns={"Filler":prefix_down}, inplace=True)

                plots_manager = LonePlotManager(metric_df=metric_df, plt_options=options, version=v_dir.stem)

                if not start_overwrite_figaxs and figax_store[i] is None:
                    figax_store[i] = plots_manager # For runs in info they transfer on the same figax
                else:
                    used_manager = figax_store[i]
                    assert type(used_manager) is LonePlotManager, ""
                    plots_manager.overwrite_figax(figax=used_manager.lone_figax)

                plots_managers.append(plots_manager)

    for i, mngr in enumerate(plots_managers):
        assert type(mngr) is LonePlotManager, ""

        fig, ax = mngr.viz_cols_in_one()
        save_dir = f"{str(outputs_dir.resolve())}/report/{mode}/{mngr.version}/"
        os.makedirs(save_dir, exist_ok=True)
        # Directory path should ensure required uniqueness settings 
        fig.savefig(f"{save_dir}/lone_{final_fig_name.replace(" ", "_").lower()}.png") 

def viz_grid_metrics(mode:str, outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str, final_fig_name:str) -> None:
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
            print(f"v_dir: {v_dir.stem} matched to idx: {i}")
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
                optimizer_name = "Muon " if "muon" in str(experim_dir).lower() else "Adam "

                prefix_down = optimizer_name.capitalize()
                prefix_down = prefix_down.replace("-", " ").replace("_", " ").capitalize()

                metric_df = pd.read_parquet(path)
                # TODO: Have plot options passed in immediately later on if needed, no need to extract it so heavily, do like fig name
                x_label = "Epochs" if "epochs" in path.stem.lower() else "Steps"
                y_label = "Unknown"
                if "accuracy" in path.stem.lower():
                    y_label = "Accuracy" 
                elif "loss" in path.stem.lower():
                    y_label = "Loss" 
                elif "unnormalised" in path.stem.lower():
                    y_label = "Dot Product"
                else:
                    y_label = "Cosine Similarity"

                options = GridPlotOptions(x_label=x_label, y_label=y_label, grid_title= optimizer_name + " " + mode + " " + final_fig_name, grid_cols_num=5, prefix=prefix_down)
                assert type(options) is GridPlotOptions

                plots_manager = GridPlotsmanager(metric_df=metric_df, plt_options=options, version=v_dir.stem)

                if not start_overwrite_figaxs and figaxs_store[i] is None:
                    figaxs_store[i] = plots_manager # For runs in info they transfer on the same figax
                else:
                    used_manager = figaxs_store[i]
                    assert type(used_manager) is GridPlotsmanager, ""
                    plots_manager.overwrite_figaxes(figaxes=used_manager.grid_figaxs)

                plots_managers.append(plots_manager)

    for i, mngr in enumerate(plots_managers): 
        assert type(mngr) is GridPlotsmanager, ""

        fig, axs = mngr.viz_cols_in_grid()
        save_dir = f"{str(outputs_dir.resolve())}/report/{mode}/{mngr.version}/"
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f"{save_dir}/grid_{final_fig_name.replace(" ", "_").lower()}.png") 
        
if __name__ == "__main__":
    """ 
    Visualize non-core metrics that were created. 
    """
    modes = ["mean", "std"]
    for mode in modes: 
        # TODO: Do this again but for STD not mean (replace or do loop for modes (easiest))
        print("=========== VIZ UNNORMED GRAD CORR METRICS FOR REPORT ===========") 
        # Unnormed Per col per step (per OPTIM step)
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-none_{mode}_Unnormalised-GC.parquet", 
            )
        ]

        viz_grid_metrics(
            mode=mode,
            outputs_dir=cls_imb_l_outputs_dir, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Per Layer Dot Product (Grid)"
        ) 

        # Unnormed Col reduction per step (per OPTIM step)
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-cols_{mode}_Unnormalised-GC.parquet", 
            )
        ]

        viz_lone_metrics(
            mode=mode,
            outputs_dir=cls_imb_l_outputs_dir, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Muon and Adam Dot Product Comparison"
        ) 

        print("=========== VIZ NORMED GRAD CORR METRICS FOR REPORT ===========") 
        # Normed Per col per step (per OPTIM step)
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-none_{mode}_Normalised-GC.parquet", 
            )
        ]

        viz_grid_metrics(
            mode=mode,
            outputs_dir=cls_imb_l_outputs_dir, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Per Layer Cosine Similarity (Grid)"
        ) 

        # Normed Col reduction per step (per OPTIM step)
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-cols_{mode}_Normalised-GC.parquet", 
            )
        ]

        viz_lone_metrics(
            mode=mode,
            outputs_dir=cls_imb_l_outputs_dir, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Muon and Adam Cosine Similarity Comparison"
        ) 

        print("=========== VIZ SUMMARY ACCURACY METRICS FOR REPORT ===========") 
        # Cols present per epoch
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-epochs_{mode}_Train-Accuracy.parquet", 
            ),
            dict(
                relative_path = f"runs_extra-epochs_{mode}_Validation-Accuracy.parquet", 
            ) 
        ]

        viz_lone_metrics(
            mode=mode,
            outputs_dir=cls_imb_l_outputs_dir, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Muon and Adam Train and Validation Accuracy (Per Class Across Epochs)"
        ) 
        
        # Col reduction per epoch
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-both_{mode}_Train-Accuracy.parquet", 
            ),
            dict(
                relative_path = f"runs_extra-both_{mode}_Validation-Accuracy.parquet", 
            ) 
        ]

        viz_lone_metrics(
            mode=mode,
            outputs_dir=cls_imb_l_outputs_dir, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Muon and Adam Train and Validation Accuracy (Per Epoch Global Average)"
        ) 

        # Subgroups per epoch
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-subgroup_epochs_{mode}_Train-Accuracy.parquet", 
            ),
            dict(
                relative_path = f"runs_extra-subgroup_epochs_{mode}_Validation-Accuracy.parquet", 
            ) 
        ]

        viz_lone_metrics(
            mode=mode,
            outputs_dir=cls_imb_l_outputs_dir, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Muon and Adam Train and Validation Accuracy (Majority and Minority Class)"
        ) 

        print("=========== VIZ SUMMARY LOSS METRICS FOR REPORT ===========") 
        # Loss per step (NOT per optim step)
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-none_{mode}_Train-Loss.parquet", 
            ),
            dict(
                relative_path = f"runs_extra-none_{mode}_Validation-Loss.parquet", 
            ) 
        ]

        viz_lone_metrics(
            mode=mode,
            outputs_dir=cls_imb_l_outputs_dir, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Muon and Adam Train and Validation Loss"
        ) 


