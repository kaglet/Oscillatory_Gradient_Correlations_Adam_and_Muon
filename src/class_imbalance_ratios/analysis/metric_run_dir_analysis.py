""" 
Main entry point for acc analysis. 
"""

from pathlib import Path
from typing import Any
from IPython.display import display

from src.base.lit_module.components.metric_gch import GCH_Metric, RGCH_Metric
from src.base.lit_module.components.metrics_standard import MyLoss, MyMultiClassAccuracy
from src.class_imbalance_ratios.analysis.summarise_metrics import summarize_runs_metric

def agg_save_metrics(outputs_dir: Path, run_paths_info: list[dict[str, Any]], version_dir: str, limit: int | None = 1, corr_store_names: dict[str, str] = dict(normed="GCH", raw="RGCH")) -> None:
    """ 
    Pass in list of patterns. and aggregation options, which are looped over for optional number of times. 
    """
    limit = 0 if limit is None else limit 

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

if __name__ == "__main__":

    pass