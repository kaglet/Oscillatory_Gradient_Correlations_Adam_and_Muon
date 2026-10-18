import math
from typing import Any, Tuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import gc

def show_plt_with_cleanup(fig: Figure) -> bool:
    plt.show()
    plt.close(fig)
    gc.collect()
    return True

def cleanup_all() -> bool:
    plt.close('all')
    return True

class LonePlotOptions():
    def __init__(self, x_label: str, y_label:str, unique_label: str, title_y_offset:float = 1.0, font_size: int = 200, dpi: int = 400, base_size: int= 8) -> None:
        self.title: str = unique_label
        self.title_y_offset = title_y_offset
        self.fontsize = font_size
        self.dpi = dpi
        self._base_size: int = base_size
        self.y_label = y_label
        self.x_label = x_label

class GridPlotOptions(LonePlotOptions):
    def __init__(self, x_label:str, y_label:str, grid_cols_num: int, grid_title: str, title_y_offset: float = 1, font_size: int = 200, dpi: int = 400, base_size: int = 8) -> None:
        super().__init__(x_label=x_label, y_label=y_label,unique_label=grid_title, title_y_offset=title_y_offset, font_size=font_size, dpi=dpi, base_size=base_size)
        self.grid_cols_num: int = grid_cols_num

class GridPlotsmanager():
    def __init__(self, metric_df: pd.DataFrame, plt_options: GridPlotOptions) -> None:
        self._metric_df = metric_df
        self._plot_options = plt_options
        self._grid_figaxs = self._create_grid_figaxes()

    @property
    def grid_figaxs(self):
        return self._grid_figaxs

    @grid_figaxs.setter
    def grid_figaxs(self, figaxs):
        self._grid_figaxs = figaxs

    def _create_grid_figaxes(self) -> Tuple[Figure, Any]:
        self._total_cols = len(self._metric_df.columns)
        self._grid_rows_num = math.ceil(self._total_cols / self._plot_options.grid_cols_num)
        fig, axs = plt.subplots(
            self._plot_options.grid_cols_num, self._grid_rows_num, 
            dpi=self._plot_options.dpi,
            figsize = ((self._plot_options._base_size + 60) * self._plot_options.grid_cols_num, (self._plot_options._base_size - 6) * self._grid_rows_num), 
            sharex=True, 
            sharey=True
        )
        figaxes = fig, axs
        return figaxes

    def viz_cols_in_grid(self) -> None:
        """

        """
        # figsize is (w,h) as spec
        fig, axs = self.grid_figaxs
        fig.suptitle(f'{self._plot_options.title.upper()}', fontsize=self._plot_options.fontsize, y=self._plot_options.title_y_offset)
        cols_limit_counter = 0
        for grid_row in range(self._plot_options.grid_cols_num):
            for grid_col in range(self._grid_rows_num):
                if cols_limit_counter >= self._total_cols:
                    break 

                col = self._metric_df.columns[cols_limit_counter]
                index_vals = list(self._metric_df.index.to_numpy(dtype=np.int32))
                col_vals = list(self._metric_df.loc[:, col].values)
                cols_limit_counter += 1
                line, = axs[grid_row, grid_col].plot(index_vals, col_vals, marker='o', markersize=2.5, linestyle='-', linewidth=0.4)

                axs[grid_row, grid_col].tick_params(axis='both', which='both', labelsize=7, labelbottom=True, labelleft=True)

                axs[grid_row, grid_col].set_ylabel(self._plot_options.title.upper()) 
                axs[grid_row, grid_col].set_xlabel('Steps', fontsize=120)           
                axs[grid_row, grid_col].set_title(f"{col.upper()}")

                axs[grid_row, grid_col].tick_params(axis='x', fontsize=70)
                axs[grid_row, grid_col].tick_params(axis='y', fontsize=70)

                axs[grid_row, grid_col].legend(loc="upper right")

    def overwrite_figaxes(self, figaxes: Tuple[Figure, Any]) -> bool:
        fig, axs = figaxes
        assert type(axs) == type(self.grid_figaxs[1]), f"Type of axs to overwrite current must be {type(self.grid_figaxs[1])} not {type(axs)}."
        rows_equal = len(self.grid_figaxs[1]) == len(axs)
        cols_equal = len(self.grid_figaxs[1][0]) == len(axs[0])
        # TODO: If uneven grid would need to expand this to be a check over all dims (easier if np.ndarray)
        if rows_equal and cols_equal:
            self._grid_figaxs = (fig, axs)
        else:
            raise ValueError()
        
        return True

class LonePlotManager():
    def __init__(self, metric_df: pd.DataFrame, plt_options: LonePlotOptions) -> None:
        self._metric_df = metric_df
        self._plot_options = plt_options
        self._lone_figax = self._create_lone_figax()

    def _create_lone_figax(self) -> Tuple[Figure, Axes]:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize = ((self._plot_options._base_size), (self._plot_options._base_size)))
        figax = fig, ax
        return figax

    @property
    def lone_figax(self):
        return self._lone_figax

    @lone_figax.setter
    def lone_figax(self, figax):
        self._grid_figaxs = figax

    def viz_cols_in_one(self) -> None:
        """
        For a metric object instance, visualize all col of `values` with its associated string in `header`.
        Internal output is figure with 1 plot for all column , labelled by col name in Index.
        Optionally save as compressed numpy file with flags per viz (Recommended).
        Optionally save figure (Recommended) per viz. 
        """
        fig, ax = self._lone_figax
        fig.suptitle(f'{self._plot_options.title.upper()}')

        for col in self._metric_df.columns:
            line, = ax.plot(list(self._metric_df.index.to_numpy(dtype=np.int32)), list(self._metric_df.loc[:, col].values), marker='o', linestyle='-')
            line.set_label(col.upper())

        # ax.tick_params(axis='both', which='both', labelsize=7, labelbottom=True, labelleft=True)

        ax.tick_params(axis='x')
        ax.tick_params(axis='y')

        ax.set_ylabel(self._plot_options.y_label.upper())
        ax.set_xlabel("Steps")

        ax.legend(loc="upper right")
        show_plt_with_cleanup(fig)

    def overwrite_figax(self, figax: Tuple[Figure, Axes]) -> bool:
        self._lone_figax = figax
        return True

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
    # viz_cols_in_grid(metric_df=metric_df)