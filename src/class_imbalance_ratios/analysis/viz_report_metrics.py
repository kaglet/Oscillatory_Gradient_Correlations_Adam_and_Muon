import os
from pathlib import Path
from typing import Any

from types import SimpleNamespace
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
from src.class_imbalance_ratios.analysis.needed_paths import cls_imb_s_outputs_dir, cls_imb_l_outputs_dir
from src.class_imbalance_ratios.analysis.plot_managers import GridPlotOptions, GridPlotsmanager, LonePlotOptions, LonePlotManager

def get_mix_match_names(mode: str, final_fig_name: str, experim_dir: Path, path: Path, v_dir: Path) -> SimpleNamespace:
    optimizer_name = "Muon" if "muon" in str(experim_dir).lower() else "Adam"

    last_underscore = str(path.stem).rfind("_") + 1
    version = str(v_dir.stem).upper()
    metric_name = str(path.stem).upper()[last_underscore: ].replace("-", " ").replace("_", " ")

    x_label = "Epochs" if "epochs" in path.stem.lower() else "Steps" # Default is steps assuming no aggregation
    y_label = "Unknown"
    if "accuracy" in path.stem.lower():
        y_label = "Accuracy" 
    elif "loss" in path.stem.lower():
        y_label = "Loss" 
    elif "unnormalised" in path.stem.lower():
        y_label = "Dot Product"
    else:
        y_label = "Cosine Similarity"

    if "train" in path.stem.lower():
        train_split = "train"
    elif "val" in path.stem.lower(): 
        train_split = "val"
    else: 
        train_split = "train"
    
    train_split = train_split.title()

    optimizer_name = optimizer_name.title()
    mode = mode.title()
    final_fig_name=final_fig_name.title()
    x_label = x_label.title()
    y_label = y_label.title()
    mix_match_names = SimpleNamespace(use_prop_cycle=cycle_on_prop_checker(path), mode=mode, unique_cols=use_unique_col_names(path), version=version, metric_name=metric_name, x_label = x_label, y_label = y_label, final_fig_name=final_fig_name, optimizer_name=optimizer_name, train_split=train_split)

    return mix_match_names

def cycle_on_prop_checker(path: Path) -> bool:
    cycle_yes = True
    if "accuracy" in path.stem.lower():
        if "subgroup_steps" in str(path.stem).lower(): # subgroup cases first
            cycle_yes = True
        elif "subgroup_epochs" in str(path.stem).lower():
            cycle_yes = True
        elif "epochs" in path.stem.lower(): # fine grained per epoch case, no subgroups detected
            cycle_yes = False 

    return cycle_yes

def use_unique_col_names(path: Path) -> bool:
    unique_cols = False
    if "accuracy" in path.stem.lower():
        if "subgroup_steps" in str(path.stem).lower(): # subgroup cases first
            unique_cols = True
        elif "subgroup_epochs" in str(path.stem).lower():
            unique_cols = True
        elif "epochs" in path.stem.lower(): # fine grained per epoch case, no subgroups detected
            unique_cols = True 

    return unique_cols

def viz_lone_metrics(mode:str, outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str, final_fig_name:str) -> None:
    """ 
    Pass in list of patterns. and aggregation options, which are looped over for optional number of times. 
    """
    figax_store: list = []
    plots_managers = []

    for experim_dir in sorted(outputs_dir.iterdir()):
        version_dirs = sorted(experim_dir.glob(version_dir))
        
        for i, v_dir in enumerate(version_dirs):
            # print(f"v_dir: {v_dir.stem} matched to idx: {i}")
            # Better for figax_store to have a fixed length based of versions
            # Then index into that when repeating diff experiments
            start_overwrite_figaxs = len(figax_store) == len(version_dirs)
            if not start_overwrite_figaxs:
                figax_store.append(None)

            for info in run_paths_info:
                path = sorted(v_dir.rglob(info["relative_path"], case_sensitive=False))[0]
                if not Path(path).exists():
                    raise LookupError(f"Path to requested metric: {path} does not exit.")

                metric_df = pd.read_parquet(path)

                names = get_mix_match_names(mode, final_fig_name, experim_dir, path, v_dir)
                metric_df.rename(columns={"Filler":names.metric_name}, inplace=True)

                options = LonePlotOptions(use_prop_cycle=names.use_prop_cycle, use_cols=names.unique_cols, version=names.version, train_split=names.train_split, final_fig_name=names.final_fig_name, mode=names.mode, optim_name=names.optimizer_name, base_size=8, x_label=names.x_label, y_label=names.y_label)
                assert type(options) is LonePlotOptions

                plots_manager = LonePlotManager(metric_df=metric_df, plt_options=options)

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
        save_dir = f"{str(outputs_dir.resolve())}/report/{mode}/{mngr._plot_options.names["version"]}/"
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
            # print(f"v_dir: {v_dir.stem} matched to idx: {i}")
            start_overwrite_figaxs = len(figaxs_store) == len(version_dirs)
            if not start_overwrite_figaxs:
                figaxs_store.append(None)

            for info in run_paths_info:
                path = sorted(v_dir.rglob(info["relative_path"], case_sensitive=False))[0]
                if not Path(path).exists():
                    raise LookupError(f"Path to requested metric: {path} does not exit.")

                metric_df = pd.read_parquet(path)

                names = get_mix_match_names(mode, final_fig_name, experim_dir, path, v_dir)

                options = GridPlotOptions(use_prop_cycle=names.use_prop_cycle, use_cols=names.unique_cols, version=names.version, train_split=names.train_split, final_fig_name=names.final_fig_name, mode=names.mode, optim_name=names.optimizer_name, base_size=8, x_label=names.x_label, y_label=names.y_label, grid_cols_num=5)
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

def viz_mean_correlation(dir_path: Path, mode: str) -> None:
        print(f"=========== VIZ {mode.upper()} NORMED GRAD CORR METRICS FOR REPORT ===========") 
        # Normed Per col per step (per OPTIM step)
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-none_{mode}_Normalised-GC.parquet", 
            )
        ]

        viz_grid_metrics(
            mode=mode,
            outputs_dir=dir_path, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Muon and Adam Per Layer Cosine Similarity"
        ) 

        # Normed Col reduction per step (per OPTIM step)
        relative_paths_info = [
            dict(
                relative_path= f"runs_extra-cols_{mode}_Normalised-GC.parquet", 
            )
        ]

        viz_lone_metrics(
            mode=mode,
            outputs_dir=dir_path, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Muon and Adam Cosine Similarity"
        ) 

def viz_dot_prod(dir_path: Path, mode: str) -> None:
    print(f"=========== VIZ {mode.upper()} UNNORMED GRAD CORR METRICS FOR REPORT ===========") 
    # Unnormed Per col per step (per OPTIM step)
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-none_{mode}_Unnormalised-GC.parquet", 
        )
    ]

    viz_grid_metrics(
        mode=mode,
        outputs_dir=dir_path, 
        version_dir="multijob/R*", 
        run_paths_info=relative_paths_info,
        final_fig_name="Muon and Adam Per Layer Dot Product"
    ) 

    # Unnormed Col reduction per step (per OPTIM step)
    relative_paths_info = [
        dict(
            relative_path= f"runs_extra-cols_{mode}_Unnormalised-GC.parquet", 
        )
    ]

    viz_lone_metrics(
        mode=mode,
        outputs_dir=dir_path, 
        version_dir="multijob/R*", 
        run_paths_info=relative_paths_info,
        final_fig_name="Muon and Adam Dot Product"
    ) 

def viz_accuracy(dir_path: Path, mode: str) -> None:
        print(f"=========== VIZ {mode.upper()} SUMMARY ACCURACY METRICS FOR REPORT ===========") 
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
            outputs_dir=dir_path, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Per Class Muon and Adam Train and Validation Accuracy"
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
            outputs_dir=dir_path, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Muon and Adam Train and Validation Accuracy (Global)"
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
            outputs_dir=dir_path, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Subgroup Muon and Adam Train and Validation Accuracy"
        ) 

def viz_loss(dir_path: Path, mode: str) -> None:
        print(f"=========== VIZ {mode.upper()} SUMMARY LOSS METRICS FOR REPORT ===========") 
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
            outputs_dir=dir_path, 
            version_dir="multijob/R*", 
            run_paths_info=relative_paths_info,
            final_fig_name="Muon and Adam Train and Validation Loss"
        ) 

def viz_over_dir(dir_path: Path) -> None:
    """ 
    Visualize non-core metrics that were created. 
    """
    MODES = ["mean", "std"]
    for mode in MODES: 
        viz_accuracy(dir_path=dir_path, mode=mode)
        viz_loss(dir_path=dir_path, mode=mode)
        viz_dot_prod(dir_path=dir_path, mode=mode)
        viz_mean_correlation(dir_path=dir_path, mode=mode)

if __name__ == "__main__":
    viz_over_dir(cls_imb_s_outputs_dir)