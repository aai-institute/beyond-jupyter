import logging
from typing import Sequence, Callable, TypeVar, Tuple, Optional, List, Any

import matplotlib.figure
import matplotlib.ticker as plticker
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

log = logging.getLogger(__name__)

MATPLOTLIB_DEFAULT_FIGURE_SIZE = (6.4, 4.8)


class Color:
    def __init__(self, c: Any):
        """
        :param c: any color specification that is understood by matplotlib
        """
        self.rgba = matplotlib.colors.to_rgba(c)

    def darken(self, amount: float):
        """
        :param amount: amount to darken in [0,1], where 1 results in black and 0 leaves the color unchanged
        :return: the darkened color
        """
        import colorsys
        rgb = matplotlib.colors.to_rgb(self.rgba)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        l *= amount
        rgb = colorsys.hls_to_rgb(h, l, s)
        return Color((*rgb, self.rgba[3]))

    def lighten(self, amount: float):
        """
        :param amount: amount to lighten in [0,1], where 1 results in white and 0 leaves the color unchanged
        :return: the lightened color
        """
        import colorsys
        rgb = matplotlib.colors.to_rgb(self.rgba)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        l += (1-l) * amount
        rgb = colorsys.hls_to_rgb(h, l, s)
        return Color((*rgb, self.rgba[3]))

    def alpha(self, opacity: float) -> "Color":
        """
        Returns a new color with modified alpha channel (opacity)
        :param opacity: the opacity between 0 (transparent) and 1 (fully opaque)
        :return: the modified color
        """
        if not (0 <= opacity <= 1):
            raise ValueError(f"Opacity must be between 0 and 1, got {opacity}")
        return Color((*self.rgba[:3], opacity))

    def to_hex(self, keep_alpha=True) -> str:
        return matplotlib.colors.to_hex(self.rgba, keep_alpha)


class LinearColorMap:
    """
    Facilitates usage of linear segmented colour maps by combining a colour map (member `cmap`), which transforms normalised values in [0,1]
    into colours, with a normaliser that transforms the original values. The member `scalarMapper`
    """
    def __init__(self, norm_min, norm_max, cmap_points: List[Tuple[float, Any]], cmap_points_normalised=False):
        """
        :param norm_min: the value that shall be mapped to 0 in the normalised representation (any smaller values are also clipped to 0)
        :param norm_max: the value that shall be mapped to 1 in the normalised representation (any larger values are also clipped to 1)
        :param cmap_points: a list (of at least two) tuples (v, c) where v is the value and c is the colour associated with the value;
            any colour specification supported by matplotlib is admissible
        :param cmap_points_normalised: whether the values in `cmap_points` are already normalised
        """
        self.norm = matplotlib.colors.Normalize(vmin=norm_min, vmax=norm_max, clip=True)
        if not cmap_points_normalised:
            cmap_points = [(self.norm(v), c) for v, c in cmap_points]
        self.cmap = LinearSegmentedColormap.from_list(f"cmap{id(self)}", cmap_points)
        self.scalarMapper = matplotlib.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_color(self, value):
        rgba = self.scalarMapper.to_rgba(value)
        return '#%02x%02x%02x%02x' % tuple(int(v * 255) for v in rgba)


def plot_matrix(matrix: np.ndarray, title: str, xtick_labels: Sequence[str], ytick_labels: Sequence[str], xlabel: str,
        ylabel: str, normalize=True, figsize: Tuple[int, int] = (9, 9), title_add: str = None) -> matplotlib.figure.Figure:
    """
    :param matrix: matrix whose data to plot, where matrix[i, j] will be rendered at x=i, y=j
    :param title: the plot's title
    :param xtick_labels: the labels for the x-axis ticks
    :param ytick_labels: the labels for the y-axis ticks
    :param xlabel: the label for the x-axis
    :param ylabel: the label for the y-axis
    :param normalize: whether to normalise the matrix before plotting it (dividing each entry by the sum of all entries)
    :param figsize: an optional size of the figure to be created
    :param title_add: an optional second line to add to the title
    :return: the figure object
    """
    matrix = np.transpose(matrix)

    if title_add is not None:
        title += f"\n {title_add} "

    if normalize:
        matrix = matrix.astype('float') / matrix.sum()
    fig, ax = plt.subplots(figsize=figsize)
    fig.canvas.manager.set_window_title(title.replace("\n", " "))
    # We want to show all ticks...
    ax.set(xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=xtick_labels, yticklabels=ytick_labels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel)
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.4f' if normalize else ('.2f' if matrix.dtype.kind == 'f' else 'd')
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    return fig


TPlot = TypeVar("TPlot", bound="Plot")


class Plot:
    def __init__(self, draw: Callable[[], None] = None, name=None):
        """
        :param draw: function which returns a matplotlib.Axes object to show
        :param name: name/number of the figure, which determines the window caption; it should be unique, as any plot
            with the same name will have its contents rendered in the same window. By default, figures are number
            sequentially.
        """
        fig, ax = plt.subplots(num=name)
        self.fig: plt.Figure = fig
        self.ax: plt.Axes = ax
        if draw is not None:
            draw()

    def xlabel(self: TPlot, label) -> TPlot:
        self.ax.set_xlabel(label)
        return self

    def ylabel(self: TPlot, label) -> TPlot:
        self.ax.set_ylabel(label)
        return self

    def title(self: TPlot, title: str) -> TPlot:
        self.ax.set_title(title)
        return self

    def xlim(self: TPlot, min_value, max_value) -> TPlot:
        self.ax.set_xlim(min_value, max_value)
        return self

    def ylim(self: TPlot, min_value, max_value) -> TPlot:
        self.ax.set_ylim(min_value, max_value)
        return self

    def save(self, path):
        log.info(f"Saving figure in {path}")
        self.fig.savefig(path)

    def xtick(self: TPlot, major=None, minor=None) -> TPlot:
        """
        Sets a tick on every integer multiple of the given base values.
        The major ticks are labelled, the minor ticks are not.

        :param major: the major tick base value
        :param minor: the minor tick base value
        :return: self
        """
        if major is not None:
            self.xtick_major(major)
        if minor is not None:
            self.xtick_minor(minor)
        return self

    def xtick_major(self: TPlot, base) -> TPlot:
        self.ax.xaxis.set_major_locator(plticker.MultipleLocator(base=base))
        return self

    def xtick_minor(self: TPlot, base) -> TPlot:
        self.ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=base))
        return self

    def ytick_major(self: TPlot, base) -> TPlot:
        self.ax.yaxis.set_major_locator(plticker.MultipleLocator(base=base))
        return self


class ScatterPlot(Plot):
    N_MAX_TRANSPARENCY = 1000
    N_MIN_TRANSPARENCY = 100
    MAX_OPACITY = 0.5
    MIN_OPACITY = 0.05

    def __init__(self, x, y, c=None, c_base: Tuple[float, float, float] = (0, 0, 1), c_opacity=None, x_label=None, y_label=None, **kwargs):
        """
        :param x: the x values; if has name (e.g. pd.Series), will be used as axis label
        :param y: the y values; if has name (e.g. pd.Series), will be used as axis label
        :param c: the colour specification; if None, compose from ``c_base`` and ``c_opacity``
        :param c_base: the base colour as (R, G, B) floats
        :param c_opacity: the opacity; if None, automatically determine from number of data points
        :param x_label:
        :param y_label:
        :param kwargs:
        """
        if c is None:
            if c_base is None:
                c_base = (0, 0, 1)
            if c_opacity is None:
                n = len(x)
                if n > self.N_MAX_TRANSPARENCY:
                    transparency = 1
                elif n < self.N_MIN_TRANSPARENCY:
                    transparency = 0
                else:
                    transparency = (n - self.N_MIN_TRANSPARENCY) / (self.N_MAX_TRANSPARENCY - self.N_MIN_TRANSPARENCY)
                c_opacity = self.MIN_OPACITY + (self.MAX_OPACITY - self.MIN_OPACITY) * (1-transparency)
            c = ((*c_base, c_opacity),)

        assert len(x) == len(y)
        if x_label is None and hasattr(x, "name"):
            x_label = x.name
        if y_label is None and hasattr(y, "name"):
            y_label = y.name

        def draw():
            if x_label is not None:
                plt.xlabel(x_label)
            if x_label is not None:
                plt.ylabel(y_label)
            plt.scatter(x, y, c=c, **kwargs)

        super().__init__(draw)


class HeatMapPlot(Plot):
    DEFAULT_CMAP_FACTORY = lambda num_points: LinearSegmentedColormap.from_list("whiteToRed",
        ((0, (1, 1, 1)), (1 / num_points, (1, 0.96, 0.96)), (1, (0.7, 0, 0))), num_points)

    def __init__(self, x, y, x_label=None, y_label=None, bins=60, cmap=None, common_range=True, diagonal=False,
            diagonal_color="green", **kwargs):
        """
        :param x: the x values
        :param y: the y values
        :param x_label: the x-axis label
        :param y_label: the y-axis label
        :param bins: the number of bins to use in each dimension
        :param cmap: the colour map to use for heat values (if None, use default)
        :param common_range: whether the heat map is to use a common rng for the x- and y-axes (set to False if x and y are completely
            different quantities; set to True use cases such as the evaluation of regression model quality)
        :param diagonal: whether to draw the diagonal line (useful for regression evaluation)
        :param diagonal_color: the colour to use for the diagonal line
        :param kwargs: parameters to pass on to plt.imshow
        """
        assert len(x) == len(y)
        if x_label is None and hasattr(x, "name"):
            x_label = x.name
        if y_label is None and hasattr(y, "name"):
            y_label = y.name

        def draw():
            nonlocal cmap
            x_range = [min(x), max(x)]
            y_range = [min(y), max(y)]
            rng = [min(x_range[0], y_range[0]), max(x_range[1], y_range[1])]
            if common_range:
                x_range = y_range = rng
            if diagonal:
                plt.plot(rng, rng, '-', lw=0.75, label="_not in legend", color=diagonal_color, zorder=2)
            heatmap, _, _ = np.histogram2d(x, y, range=[x_range, y_range], bins=bins, density=False)
            extent = [x_range[0], x_range[1], y_range[0], y_range[1]]
            if cmap is None:
                cmap = HeatMapPlot.DEFAULT_CMAP_FACTORY(len(x))
            if x_label is not None:
                plt.xlabel(x_label)
            if y_label is not None:
                plt.ylabel(y_label)
            plt.imshow(heatmap.T, extent=extent, origin='lower', interpolation="none", cmap=cmap, zorder=1, aspect="auto", **kwargs)

        super().__init__(draw)


class HistogramPlot(Plot):
    def __init__(self, values, bins="auto", kde=False, cdf=False, cdf_complementary=False, cdf_secondary_axis=True,
            binwidth=None, stat="probability", xlabel=None,
            **kwargs):
        """
        :param values: the values to plot
        :param bins: a bin specification as understood by sns.histplot
        :param kde: whether to add a kernel density estimator
        :param cdf: whether to add a plot of the cumulative distribution function (cdf)
        :param cdf_complementary: whether to plot, if cdf is enabled, the complementary values
        :param cdf_secondary_axis: whether to use, if cdf is enabled, a secondary
        :param binwidth: the bin width; if None, inferred
        :param stat: the statistic to plot (as understood by sns.histplot)
        :param xlabel: the label for the x-axis
        :param kwargs: arguments to pass on to sns.histplot
        """

        def draw():
            nonlocal cdf_secondary_axis
            sns.histplot(values, bins=bins, kde=kde, binwidth=binwidth, stat=stat, **kwargs)
            plt.ylabel(stat)
            if cdf:
                ecdf_stat = stat
                if ecdf_stat not in ("count", "proportion", "probability"):
                    ecdf_stat = "proportion"
                    cdf_secondary_axis = True
                cdf_ax: Optional[plt.Axes] = None
                cdf_ax_label = f"{ecdf_stat} (cdf)"
                if cdf_secondary_axis:
                    cdf_ax: plt.Axes = plt.twinx()
                    if stat in ("proportion", "probability"):
                        y_tick = 0.1
                    elif stat == "percent":
                        y_tick = 10
                    else:
                        y_tick = None
                    if y_tick is not None:
                        cdf_ax.yaxis.set_major_locator(plticker.MultipleLocator(base=y_tick))
                if cdf_complementary or ecdf_stat in ("count", "proportion", "probability"):
                    ecdf_stat = "proportion" if stat == "probability" else stat  # same semantics but "probability" not understood by ecdfplot
                    sns.ecdfplot(values, stat=ecdf_stat, complementary=cdf_complementary, color="orange", ax=cdf_ax)
                else:
                    sns.histplot(values, bins=100, stat=stat, element="poly", fill=False, cumulative=True, color="orange", ax=cdf_ax)
                if cdf_ax is not None:
                    cdf_ax.set_ylabel(cdf_ax_label)
            if xlabel is not None:
                self.xlabel(xlabel)

        super().__init__(draw)
