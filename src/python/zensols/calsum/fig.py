"""A simple object oriented plotting API.

"""
from __future__ import annotations
__author__ = 'Paul Landes'
from typing import Tuple, Dict, Any, Union, Type, Callable
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.figure import Figure as MatplotFigure
from zensols.persist import (
    persisted, PersistedWork, FileTextUtil, Deallocatable
)
from zensols.config import Dictable, ConfigFactory

logger = logging.getLogger(__name__)


@dataclass
class Plot(Dictable, metaclass=ABCMeta):
    """An abstract base class for plots.  The subclass overrides :meth:`plot` to
    generate the plot.  Then the client can use :meth:`save` or :meth:`render`
    it.  The plot is created as a subplot providing attributes for space to be
    taken in rows, columns, height and width.

    """
    title: str = field(default=None)
    """The title to render in the plot."""

    row: int = field(default=0)
    """The row grid position of the plot."""

    column: int = field(default=0)
    """The column grid position of the plot."""

    post_hooks: Tuple[Callable, ...] = field(default=())
    """Callables to run after rendering"""

    @abstractmethod
    def _render(self, axes: Axes):
        pass

    def render(self, axes: Axes):
        if self.title is not None:
            axes.set_title(self.title)
        self._render(axes)
        for hook in self.post_hooks:
            hook(self, axes)


@dataclass
class Figure(Deallocatable):
    """An object oriented class to manage :class:`matplit.figure.Figure` and
    subplots (:class:`matplit.pyplot.Axes`).

    """
    name: str = field(default='Untitled')
    """Used for file naming and the title."""

    config_factory: ConfigFactory = field(default=None)
    """The configuration factory used to create plots."""

    title_font_size: int = field(default=0)
    """The font size :obj:`title`.  A size of 0 means do not render the title.
    Typically a font size of 16 is appropriate.

    """
    height: int = field(default=5)
    """The height of the subplot."""

    width: int = field(default=5)
    """The width of the subplot."""

    padding: float = field(default=5.)
    """Tight layout padding."""

    plots: Tuple[Plot] = field(default=())
    """The plots managed by this object instance.  Use :meth:`add_plot` to add
    new plots.

    """
    image_dir: Path = field(default=Path('.'))
    """Where the images are stored."""

    image_format: str = field(default='svg')
    """The image format to use when saving plots."""

    def __post_init__(self):
        super().__init__()
        self._subplots = PersistedWork('_subplots', self)

    def add_plot(self, plot: Plot):
        """Add to the collection of managed plots.  This is needed for the plot
        to work if not created from this manager instance.

        :param plot: the plot to be managed

        """
        self.plots = (*self.plots, plot)

    def create(self, name: Union[str, Type[Plot]], **kwargs) -> Plot:
        """Create a plot using the arguments of :class:`.Plot`.

        :param name: the configuration section name of the plot

        :param kwargs: the initializer keyword arguments when creating the plot

        """
        plot: Plot
        if isinstance(name, Type):
            plot = name(**kwargs)
        else:
            plot = self.config_factory.new_instance(name, **kwargs)
        self.add_plot(plot)
        return plot

    @persisted('_subplots')
    def _get_subplots(self) -> Axes:
        """The subplot matplotlib axes.  A new subplot is create on the first
        time this is accessed.

        """
        fig, axs = plt.subplots(
            ncols=max(map(lambda p: p.column, self.plots)) + 1,
            nrows=max(map(lambda p: p.row, self.plots)) + 1,
            figsize=(self.width, self.height))
        fig.tight_layout(pad=self.padding)
        if self.title_font_size > 0:
            fig.suptitle(self.name, fontsize=self.title_font_size)
        return fig, axs

    def _get_axes(self) -> Union[Axes, np.ndarray]:
        return self._get_subplots()[1]

    def _get_figure(self) -> MatplotFigure:
        """The matplotlib figure."""
        return self._get_subplots()[0]

    @property
    def path(self) -> Path:
        """Where to save the image."""
        file_name: str = FileTextUtil.normalize_text(self.name)
        file_name = f'{file_name}.{self.image_format}'
        return self.image_dir / file_name

    def _get_image_metadata(self) -> Dict[str, Any]:
        return {'Title': self.name}

    def save(self) -> Path:
        """Save the figure of subplot(s) to at location :obj:`path`.

        :return: the value of :obj:`path`

        """
        axes: Union[Axes, np.ndarray] = self._get_axes()
        path: Path = self.path
        plot: Plot
        for plot in self.plots:
            ax: Axes = axes
            if isinstance(ax, np.ndarray):
                if len(ax.shape) == 1:
                    ix = plot.row if plot.row != 0 else plot.column
                    ax = axes[ix]
                else:
                    ax = axes[plot.row, plot.column]
            plot.render(ax)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._get_figure().savefig(
            fname=path,
            format=self.image_format,
            bbox_inches='tight',
            metadata=self._get_image_metadata())
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'wrote: {path}')
        self._saved = True
        return path

    def clear(self):
        """Remove all plots."""
        if self._subplots.is_set():
            fig: MatplotFigure = self._get_figure()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'deallocating fig: {fig}')
            fig.clear()
        self._subplots.clear()
        self.plots = ()

    def deallocate(self):
        self.clear()
