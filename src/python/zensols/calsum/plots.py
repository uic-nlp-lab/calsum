"""Figure plotters.

"""
__author__ = 'Paul Landes'

from typing import Tuple, Dict, Any
from dataclasses import dataclass, field
import pandas as pd
from matplotlib.pyplot import Axes
from .fig import Plot

# TODO: LossPlot will replace zensols.deeplearn.result.ModelResultGrapher


@dataclass
class LossPlot(Plot):
    losses: Tuple[Tuple[str, pd.DataFrame], ...] = field(default=())
    sample_rate: int = field(default=0)
    loss_column: str = field(default='loss')
    epoch_column: str = field(default='epoch')

    def _render(self, axes: Axes):
        import seaborn as sns
        hue_name: str = 'Embeddings Model'
        epoch_name: str = 'Epoch'
        loss_name: str = 'Loss'
        df: pd.DataFrame = None
        desc: str
        dfl: pd.DataFrame
        for desc, dfl in self.losses:
            dfl = dfl[[self.loss_column, self.epoch_column]]
            dfl = dfl.rename(columns={self.loss_column: desc})
            if df is None:
                df = dfl
            else:
                df = df.merge(
                    dfl, left_on=self.epoch_column, right_on=self.epoch_column,
                    suffixes=(None, None))
        if self.sample_rate > 0:
            df = df[(df.index % self.sample_rate) == 0]
        df = df.rename(columns={self.epoch_column: epoch_name})
        df = df.melt(epoch_name, var_name=hue_name, value_name=loss_name)
        sns.pointplot(ax=axes, data=df, x=epoch_name, y=loss_name, hue=hue_name,
                      palette='r g b m r g b m'.split(),
                      markersize=0,
                      linewidth=1.5,
                      linestyles='- - - - -- -- -- --'.split())


@dataclass
class HeatMapPlot(Plot):
    """Create an annotation heat map for all windows and optionally normalize.

    """
    dataframe: pd.DataFrame = field(default=None)
    format: str = field(default='.2f')
    x_label_rotation: float = field(default=0)
    params: Dict[str, Any] = field(default_factory=dict)

    def _render(self, axes: Axes):
        import seaborn as sns
        chart = sns.heatmap(ax=axes, data=self.dataframe,
                            annot=True, fmt=self.format, **self.params)
        if self.x_label_rotation != 0:
            axes.set_xticklabels(
                chart.get_xticklabels(),
                rotation=self.x_label_rotation)
