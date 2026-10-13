import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

def viz_each_col_sized(metadata_df: pd.DataFrame, metric_df: pd.DataFrame, path_down_prefix: str = "Train Accuracy") -> None:
    """
    For a metric object instance, visualize all col of `values` with its associated string in `header`.
    Internal output is figure with 1 plot for all column , labelled by col name in Index.
    Optionally save as compressed numpy file with flags per viz (Recommended).
    Optionally save figure (Recommended) per viz. 
    
    * `path_down_prefix` is a continuously appended prefix passed down until the end, in terms of desired known specificity.
    """
    grid_cols_num = 4
    total_cols = len(metric_df.columns)
    grid_rows_num = math.ceil(total_cols / grid_cols_num)
    fig, axs = plt.subplots(grid_cols_num, grid_rows_num, sharex=True, sharey=True)
    fig.suptitle(f'{path_down_prefix}', fontsize=16)
    cols_limit_counter = 0
    for grid_row in range(grid_cols_num):
        for grid_col in range(grid_rows_num):
            if cols_limit_counter >= total_cols:
                break 

            col = metric_df.columns[cols_limit_counter]
            index_vals = list(metric_df.index.to_numpy(dtype=np.int32))
            col_vals = list(metric_df.loc[:, col].values)
            cols_limit_counter += 1
            line, = axs[grid_row, grid_col].plot(index_vals, col_vals)
            line.set_label(col)
    
            axs[grid_row, grid_col].set_title(f"{col}")

            axs[grid_row, grid_col].tick_params(axis='x')
            axs[grid_row, grid_col].tick_params(axis='y')

            axs[grid_row, grid_col].legend()
    
    plt.show()

def viz_each_col_unsized(metadata_df: pd.DataFrame, metric_df: pd.DataFrame, path_down_prefix: str = "Train Accuracy") -> None:
    """
    For a metric object instance, visualize all col of `values` with its associated string in `header`.
    Internal output is figure with 1 plot for all column , labelled by col name in Index.
    Optionally save as compressed numpy file with flags per viz (Recommended).
    Optionally save figure (Recommended) per viz. 
    
    * `path_down_prefix` is a continuously appended prefix passed down until the end, in terms of desired known specificity.
    """
    grid_cols_num = 4
    total_cols = len(metric_df.columns)
    grid_rows_num = math.ceil(total_cols / grid_cols_num)
    fig, axs = plt.subplots(grid_cols_num, grid_rows_num, sharex=True, sharey=True)
    fig.suptitle(f'{path_down_prefix}', fontsize=200)
    cols_limit_counter = 0
    for grid_row in range(grid_cols_num):
        for grid_col in range(grid_rows_num):
            if cols_limit_counter >= total_cols:
                break 

            col = metric_df.columns[cols_limit_counter]
            index_vals = list(metric_df.index.to_numpy(dtype=np.int32))
            col_vals = list(metric_df.loc[:, col].values)
            cols_limit_counter += 1
            line, = axs[grid_row, grid_col].plot(index_vals, col_vals)
            line.set_label(col)
    
            axs[grid_row, grid_col].set_title(f"{col}", fontsize=200)

            axs[grid_row, grid_col].tick_params(axis='x', labelsize=70)
            axs[grid_row, grid_col].tick_params(axis='y', labelsize=70)

            axs[grid_row, grid_col].legend()
    
    plt.show()

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
        line, = ax.plot(list(metric_df.index.to_numpy(dtype=np.int32)), list(metric_df.loc[:, col].values))
        line.set_label(col)
    ax.set_title(f"{path_down_prefix} Plot")

    ax.tick_params(axis='x')
    ax.tick_params(axis='y')

    ax.set_ylabel(path_down_prefix)
    ax.set_ylabel(path_down_prefix)

    ax.legend()
    plt.show()

def viz_all_col_sized(metric_df: pd.DataFrame, metadata_df: pd.DataFrame, path_down_prefix: str = "Train Accuracy") -> None:
    """
    For a metric object instance, visualize all col of `values` with its associated string in `header`.
    Internal output is figure with 1 plot for all column , labelled by col name in Index.
    Optionally save as compressed numpy file with flags per viz (Recommended).
    Optionally save figure (Recommended) per viz. 
    """
    n_cols = len(metric_df.columns)
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(n_cols, n_cols))
    for col in metric_df.columns:
        line, = ax.plot(list(metric_df.index.to_numpy(dtype=np.int32)), list(metric_df.loc[:, col].values))
        line.set_label(col)
    ax.set_title(f"{path_down_prefix} Plot", fontsize=200)

    ax.tick_params(axis='x', labelsize=70)
    ax.tick_params(axis='y', labelsize=70)

    ax.set_ylabel(path_down_prefix, fontsize=120)
    ax.set_ylabel(path_down_prefix, fontsize=120)

    ax.legend(fontsize=50, title_fontsize=90)
    plt.show()

if __name__ == "__main__":
    metric_df = pd.DataFrame(
        {
            "col1": np.random.random(100), 
            "col2": np.random.random(100),
            "col3": np.random.random(100),
            "col4": np.random.random(100),
            "col5": np.random.random(100),
            "col6": np.random.random(100),            
            "col7": np.random.random(100), 
            "col8": np.random.random(100),
            "col9": np.random.random(100),
            "col10": np.random.random(100),
            "col11": np.random.random(100),
            "col12": np.random.random(100),            
            "col13": np.random.random(100), 
            "col14": np.random.random(100),
            "col15": np.random.random(100),
            "col16": np.random.random(100),
            "col17": np.random.random(100),
            "col18": np.random.random(100),
        }
    )
    # viz_each_col_unsized(metric_df=metric_df, metadata_df=metric_df)
    viz_each_col_sized(metric_df=metric_df, metadata_df=metric_df)