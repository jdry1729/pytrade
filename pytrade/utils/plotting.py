from typing import Optional, List, Any, Iterable, Union, Tuple, Dict

import matplotlib.axes
import numpy as np
import pandas as pd
from matplotlib.ticker import Locator


def plot(ax, data, title: Optional[str] = None,
         xaxis_locator: Optional[Locator] = None,
         hlines: Optional[List[Any]] = None):
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.plot(data, alpha=0.7)
    ax.set_title(title, loc="left")
    if isinstance(data, pd.DataFrame):
        ax.legend(data.columns)
    if xaxis_locator is not None:
        ax.xaxis.set_major_locator(xaxis_locator)
    ax.ticklabel_format(axis="y", useOffset=False)
    # TODO: deprecate hlines?
    if hlines is not None:
        for hline in hlines:
            ax.axhline(hline, color="black", linestyle="--")
    return ax


def bool_change_points(series: pd.Series):
    return series[(series != series.shift()) |
                  (series.isnull() != series.isnull().shift())].dropna()


def align_yaxis(ax1, v1, ax2, v2):
    """
    Adjusts ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1.
    """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + dy, maxy + dy)


def plot_intervals(intervals, ax: Optional[matplotlib.axes.Axes] = None,
                   color: Optional[Union[str, Dict]] = "lightgrey", alpha: float = 0.8):
    for j in range(len(intervals) - 1):
        interval = intervals.iloc[j]

        interval_color = None
        if isinstance(color, str):
            interval_color = color
        elif isinstance(color, Dict) and "value" in interval:
            interval_color = color.get(interval["value"])

        ax.axvspan(interval["start"], interval["end"], alpha=alpha,
                   color=interval_color)


def plot_threshold_spans(series: pd.Series, thresholds: Iterable,
                         colors: Union[List, Tuple],
                         ax: Optional[matplotlib.axes.Axes] = None,
                         alpha: float = 0.5) -> None:
    """

    Parameters
    ----------
    series
        Series.
    thresholds
        Thresholds to use.
    colors
        Colors for the regions defined by thresholds. Should have length
        one greater than the number of thresholds.
    ax
        Axis.
    alpha
        Span opacity.

    Returns
    -------
    None
    """
    # TODO: interpolate between points crossing threshold
    thresholds = [-np.inf, *thresholds, np.inf]
    for i in range(len(thresholds) - 1):
        mask = (thresholds[i] <= series) & (series < thresholds[i + 1])
        changes = bool_change_points(mask)
        changes[mask.index[-1]] = mask.iloc[-1]
        idx = changes.index
        for j in range(len(changes) - 1):
            if changes.iloc[j]:
                ax.axvspan(idx[j], idx[j + 1], alpha=alpha, color=colors[i])
