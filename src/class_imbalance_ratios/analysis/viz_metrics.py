import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

def viz_all_col_unsized(metadata_df: pd.DataFrame, metric_df: pd.DataFrame, path_down_prefix: str = "Train Accuracy") -> None:
    """
    For a metric object instance, visualize all col of `values` with its associated string in `header`.
    Internal output is figure with 1 plot for all column , labelled by col name in Index.
    Optionally save as compressed numpy file with flags per viz (Recommended).
    Optionally save figure (Recommended) per viz. 
    
    * `path_down_prefix` is a continuously appended prefix passed down until the end, in terms of desired known specificity.
    """
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    for col in metric_df.columns:
        # display(list(metric_df.index.get_level_values(1)))
        # display(metric_df.loc[:, col].describe())
        # display(metric_df.loc[:, col].max())
        line, = ax.plot(list(metric_df.index.to_numpy(dtype=np.int32)), list(metric_df.loc[:, col].values))
        line.set_label(col)
    ax.set_title(f"{path_down_prefix} Plot")

    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    ax.set_ylabel(path_down_prefix)
    ax.set_ylabel(path_down_prefix)

    ax.legend()
    plt.show()

def viz_all_col_sized(metric_df: pd.DataFrame, metadata_df: pd.DataFrame, unified_name: str = "Train Accuracy") -> None:
    """
    For a metric object instance, visualize all col of `values` with its associated string in `header`.
    Internal output is figure with 1 plot for all column , labelled by col name in Index.
    Optionally save as compressed numpy file with flags per viz (Recommended).
    Optionally save figure (Recommended) per viz. 
    """
    n_cols = len(metric_df.columns)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(n_cols, n_cols))
    for col in metric_df.columns:
        # display(list(metric_df.index.get_level_values(1)))
        # display(metric_df.loc[:, col].describe())
        # display(metric_df.loc[:, col].max())
        line, = ax.plot(list(metric_df.index.to_numpy(dtype=np.int32)), list(metric_df.loc[:, col].values))
        line.set_label(col)
    ax.set_title(f"{unified_name} Plot", fontsize=200)

    ax.tick_params(axis='x', labelsize=70)
    ax.tick_params(axis='y', labelsize=70)

    ax.set_ylabel(unified_name, fontsize=120)
    ax.set_ylabel(unified_name, fontsize=120)

    ax.legend(fontsize=50, title_fontsize=90)
    plt.show()

if __name__ == "__main__":
    pass