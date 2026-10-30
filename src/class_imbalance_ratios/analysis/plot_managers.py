import math
from typing import Any, Tuple

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import gc
import matplotlib as mpl
import matplotlib.ticker as ticker

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class LonePlotOptions():
    def __init__(self, use_prop_cycle: bool, use_cols: bool, train_split: str, version:str, x_label: str, y_label:str, final_fig_name: str, mode: str, optim_name:str, title_y_offset:float = 0.01, font_size: int = 200, dpi: int = 400, base_size: int= 4) -> None:
        self.title_y_offset = title_y_offset
        self.fontsize = font_size
        self.dpi = dpi
        self._base_size: int = base_size
        self._use_cols = use_cols
        self._use_prop_cycle = use_prop_cycle
        self.names = dict(train_split=train_split, x_label = x_label, y_label = y_label, optim_name=optim_name, mode=mode, final_fig_name=final_fig_name, version=version)

class GridPlotOptions(LonePlotOptions):
    def __init__(self, use_prop_cycle: bool, use_cols: bool, train_split:str, version:str, final_fig_name: str, mode: str, optim_name:str, x_label:str, y_label:str, grid_cols_num: int, title_y_offset: float = 1, font_size: int = 200, dpi: int = 400, base_size: int = 8) -> None:
        super().__init__(use_prop_cycle=use_prop_cycle, use_cols=use_cols, train_split=train_split, version=version, optim_name=optim_name, mode=mode, final_fig_name=final_fig_name, x_label=x_label, y_label=y_label, title_y_offset=title_y_offset, font_size=font_size, dpi=dpi, base_size=base_size)
        self.grid_cols_num: int = grid_cols_num

class LonePlotManager():
    def __init__(self, metric_df: pd.DataFrame, plt_options: LonePlotOptions) -> None:
        self._metric_df = metric_df
        self._plot_options = plt_options
        self._lone_figax = self._create_lone_figax()

    def _create_lone_figax(self) -> Tuple[Figure, Axes]:
        fig = Figure(figsize = ((self._plot_options._base_size + 7), (self._plot_options._base_size + 7)))
        ax = fig.subplots(1, 1, sharex=True, sharey=True)
        NUM_PROPS_IN_CYCLE = 4
        if self._plot_options._use_prop_cycle:
            ax.set_prop_cycle(color=['#CC79A7', '#CC79A7', '#0072B2', '#0072B2'], linestyle=['-', '--'] * int(NUM_PROPS_IN_CYCLE / 2), marker=['.'] * NUM_PROPS_IN_CYCLE, markersize=[2.5] * NUM_PROPS_IN_CYCLE, linewidth=[1.25] * NUM_PROPS_IN_CYCLE)
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
        title = self._plot_options.names["mode"]  + " " + self._plot_options.names["final_fig_name"]
        fig.suptitle(f'{title}', fontsize="xx-large", y=0.92)

        for col in self._metric_df.columns:
            line, = ax.plot(list(self._metric_df.index.to_numpy(dtype=np.int32)), list(self._metric_df.loc[:, col].values))

            og_col = col
            col = f'{self._plot_options.names["optim_name"]} {self._plot_options.names["train_split"]}'
            if self._plot_options._use_cols:
                col = col + " " + og_col
            line.set_label(col.title())
            ax.legend(fontsize='medium')
        
        ax.grid(axis='both', color='gray', linestyle='--', linewidth=1, alpha=0.65, which="major")

        ax.set_ylabel(self._plot_options.names["y_label"].title(), fontsize='x-large')
        ax.set_xlabel(self._plot_options.names["x_label"].title(), fontsize='x-large')

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

        NUM_PROPS_IN_CYCLE = 2
        for grid_row in range(self._grid_rows_num):
            for grid_col in range(self._plot_options.grid_cols_num):
                ax = axs[grid_row, grid_col]
                if self._plot_options._use_prop_cycle:
                    ax.set_prop_cycle(color=['#CC79A7', '#0072B2'], marker=['.', '.'], markersize=[2.5] * NUM_PROPS_IN_CYCLE, linestyle=['-']*NUM_PROPS_IN_CYCLE, linewidth=[1.5]*NUM_PROPS_IN_CYCLE)

        figaxes = fig, axs
        return figaxes

    def viz_cols_in_grid(self) -> Tuple[Figure, Any]:
        """

        """
        # figsize is (w,h) as spec
        fig, axs = self.grid_figaxs
        title = self._plot_options.names["mode"]  + " " + self._plot_options.names["final_fig_name"] + " " + self._plot_options.names["x_label"]

        fig.suptitle(f'{title}', fontsize="xx-large", y=0.92)

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

                ax.set_ylabel(self._plot_options.names["y_label"].title(), fontsize='medium') 
                ax.set_xlabel(self._plot_options.names["x_label"].title(), fontsize='medium')           
                ax.set_title(f"{col.title()}", fontsize='large')

                line, = ax.plot(index_vals, col_vals, label=f'{self._plot_options.names["optim_name"]}')

                ax.grid(axis='both', color='gray', linestyle='--', linewidth=1, alpha=0.65, which="major")
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