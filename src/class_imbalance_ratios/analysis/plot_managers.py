import math
from typing import Any, Tuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import gc
import matplotlib as mpl

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class LonePlotOptions():
    def __init__(self, prefix:str, x_label: str, y_label:str, lone_title: str, title_y_offset:float = 0.01, font_size: int = 200, dpi: int = 400, base_size: int= 4) -> None:
        self.title: str = lone_title
        self.title_y_offset = title_y_offset
        self.fontsize = font_size
        self.dpi = dpi
        self._base_size: int = base_size
        self.y_label = y_label
        self.x_label = x_label
        self.prefix = prefix

class GridPlotOptions(LonePlotOptions):
    def __init__(self, prefix:str, x_label:str, y_label:str, grid_cols_num: int, grid_title: str, title_y_offset: float = 1, font_size: int = 200, dpi: int = 400, base_size: int = 8) -> None:
        super().__init__(prefix=prefix, x_label=x_label, y_label=y_label,lone_title=grid_title, title_y_offset=title_y_offset, font_size=font_size, dpi=dpi, base_size=base_size)
        self.grid_cols_num: int = grid_cols_num

class LonePlotManager():
    def __init__(self, metric_df: pd.DataFrame, plt_options: LonePlotOptions, version:str) -> None:
        self._metric_df = metric_df
        self._plot_options = plt_options
        self._lone_figax = self._create_lone_figax()
        self.version = version

    def _create_lone_figax(self) -> Tuple[Figure, Axes]:
        fig = Figure(figsize = ((self._plot_options._base_size + 7), (self._plot_options._base_size + 7)))
        ax = fig.subplots(1, 1, sharex=True, sharey=True)
        # The cycle does Muon (2 [line] plots) then Adam (2 plots), we have made proficiency for up to 8 lines on a plot at once
        ax.set_prop_cycle(color=['#CC79A7', '#E69F00', '#0072B2', '#56B4E9', '#CC79A7', '#E69F00', '#0072B2', '#56B4E9'], marker=['o', '.', 'o', '.', 'D', 's', 'D', 's'], markersize=[7] * 8, linewidth=[4] * 8)
        ax.tick_params(axis='both', labelsize='large')
        
        figax = fig, ax
        return figax

    @property
    def lone_figax(self):
        return self._lone_figax

    @lone_figax.setter
    def lone_figax(self, figax):
        self._lone_figax = figax

    def viz_cols_in_one(self) -> Tuple[Figure, Axes]:
        """
        For a metric object instance, visualize all col of `values` with its associated string in `header`.
        Internal output is figure with 1 plot for all column , labelled by col name in Index.
        Optionally save as compressed numpy file with flags per viz (Recommended).
        Optionally save figure (Recommended) per viz. 
        """
        fig, ax = self.lone_figax
        focal_prefix = self._plot_options.title.capitalize()
        fig.suptitle(f'{focal_prefix}', fontsize="xx-large", y=0.92)

        for col in self._metric_df.columns:
            line, = ax.plot(list(self._metric_df.index.to_numpy(dtype=np.int32)), list(self._metric_df.loc[:, col].values), label=f'{self._plot_options.prefix} {col}')
            ax.legend(fontsize='medium', bbox_to_anchor=(1.05, 1.0))
            ax.grid(axis='x', color='gray', linestyle=':', linewidth=1, alpha=0.5)

        ax.set_ylabel(self._plot_options.y_label.capitalize(), fontsize='x-large')
        ax.set_xlabel(self._plot_options.x_label.capitalize(), fontsize='x-large')

        return fig, ax

    def overwrite_figax(self, figax: Tuple[Figure, Axes]) -> bool:
        self.lone_figax = figax
        return True

class GridPlotsmanager():
    def __init__(self, metric_df: pd.DataFrame, plt_options: GridPlotOptions, version: str) -> None:
        self._metric_df = metric_df
        self._plot_options = plt_options
        self._grid_figaxs = self._create_grid_figaxes()
        self.version = version # This is simply to get the semantic version for experiment - bad practice as coupled with external code, will decouple into separate list

    @property
    def grid_figaxs(self):
        return self._grid_figaxs

    @grid_figaxs.setter
    def grid_figaxs(self, figaxs):
        self._grid_figaxs = figaxs

    def _create_grid_figaxes(self) -> Tuple[Figure, Any]:
        self._total_cols = len(self._metric_df.columns)
        self._grid_rows_num = math.ceil(self._total_cols / self._plot_options.grid_cols_num)
        fig = Figure(figsize=(((self._plot_options._base_size + 6) * self._plot_options.grid_cols_num, (self._plot_options._base_size - 2) * self._grid_rows_num) ))
        fig.tight_layout()
        canvas = FigureCanvas(fig)
        axs = fig.subplots(
            ncols=self._plot_options.grid_cols_num,
            nrows=self._grid_rows_num,
            sharex=True,
            sharey=True
            # dpi=self._plot_options.dpi,
        )
        fig.subplots_adjust(top=0.90)

        for grid_row in range(self._grid_rows_num):
            for grid_col in range(self._plot_options.grid_cols_num):
                axs[grid_row, grid_col].set_prop_cycle(color=['#CC79A7', '#0072B2'], marker=['o', 'o'], markersize=[3] * 2, linestyle=['-']*2, linewidth=[1.5]*2)

        figaxes = fig, axs
        return figaxes

    def viz_cols_in_grid(self) -> Tuple[Figure, Any]:
        """

        """
        # figsize is (w,h) as spec
        fig, axs = self.grid_figaxs
        focal_prefix = self._plot_options.title.capitalize()

        fig.suptitle(f'{focal_prefix}', fontsize="xx-large", y=0.92)

        cols_limit_counter = 0
        for grid_col in range(self._plot_options.grid_cols_num):
            for grid_row in range(self._grid_rows_num):
                if cols_limit_counter >= self._total_cols:
                    break 

                col = self._metric_df.columns[cols_limit_counter]
                index_vals = list(self._metric_df.index.to_numpy(dtype=np.int32))
                col_vals = list(self._metric_df.loc[:, col].values)
                cols_limit_counter += 1

                ax = axs[grid_row, grid_col]
                ax.tick_params(axis='both', which='both', labelbottom=True, labelleft=True, labelsize='x-small')

                ax.set_ylabel(self._plot_options.y_label.capitalize(), fontsize='medium') 
                ax.set_xlabel(self._plot_options.x_label.capitalize(), fontsize='medium')           
                ax.set_title(f"{col.capitalize()}", fontsize='large')
                ax.grid(axis='x', color='gray', linestyle=':', linewidth=1, alpha=0.5)

                line, = ax.plot(index_vals, col_vals, label=f'{self._plot_options.prefix} {col}')

                ax.legend(fontsize='x-small')

        # fig.show()
        return fig, axs
            
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

if __name__ == "__main__":
    mpl.rcParams['font.size'] = 150

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