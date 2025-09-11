import math
from typing import Union, Optional, Tuple, Iterable, Dict

import matplotlib.axes
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.transforms import blended_transform_factory
from pytrade.portfolio.analysis import compute_large_return_periods
from pytrade.utils.collections import is_iterable_of
from pytrade.utils.pandas import raise_if_index_not_equal


def regplot(X: pd.Series, y: pd.Series, ax: Optional[matplotlib.axes.Axes] = None):
    X.name = "x"
    if ax is None:
        fig, ax = plt.subplots()
    y = y.reindex(X.index)
    nan_mask = X.isnull() | y.isnull()
    X = X[~nan_mask]
    y = y[~nan_mask]
    # TODO: avoid statsmodels
    model = sm.OLS(y, sm.add_constant(X), missing="drop")
    fit = model.fit()
    alpha = fit.params["const"]
    beta = fit.params["x"]
    p_value = fit.pvalues["x"]
    ax.scatter(X, y, alpha=0.5)
    line_X = np.linspace(X.min(), X.max(), 2)
    line_y = alpha + beta * line_X
    ax.plot(line_X, line_y, c="r", alpha=0.8)
    ax.text(.05, .92, f"$\\beta={beta:.4f}~({p_value:.6f})$", ha='left',
            va='top', transform=ax.transAxes,
            bbox=dict(facecolor="#eee", alpha=0.5))


def horizon_plot(signal: pd.DataFrame, returns: pd.DataFrame,
                 horizons: Iterable[int] = (1, 5, 10),
                 groups: Optional[Dict] = None, log: bool = True):
    if is_iterable_of([signal, returns], pd.Series):
        signal = signal.to_frame()
        returns = returns.to_frame()

    raise_if_index_not_equal(signal.index, returns.index)
    raise_if_index_not_equal(signal.columns, returns.columns)

    assets = signal.columns
    group_titles = True
    if groups is None:
        group_titles = False
        groups = {"default": assets}

    horizons = tuple(horizons)
    group_names = tuple(groups.keys())

    N = len(horizons)
    K = len(group_names)
    log_returns = np.log(returns + 1)

    fig, axes = plt.subplots(figsize=(N * 3, 2.5 * K), nrows=K, ncols=N,
                             sharex=True, sharey=True)
    if K == 1:
        axes = np.expand_dims(axes, 0)
    for j, horizon in enumerate(horizons):
        signal_ = signal.shift(horizon)
        horizon_returns = log_returns.rolling(horizon).sum()
        if not log:
            horizon_returns = np.exp(log_returns) - 1
        for i in range(K):
            assets_ = groups[group_names[i]]
            regplot(signal_[assets_].stack(future_stack=True),
                    horizon_returns[assets_].stack(future_stack=True),
                    ax=axes[i, j])
            title = ""
            if group_titles:
                title += f"{group_names[i]}, "
            title += f"h={horizon}"
            axes[i, j].set_title(title, loc="left")
    plt.tight_layout()
    plt.show()


def dsw_plot(data: Union[pd.DataFrame, pd.Series], signal: pd.Series,
             weights: pd.Series, price: Optional[pd.Series] = None, *,
             large_return_window=50,
             large_return_threshold: float = 0.05,
             num_large_return_periods=3, figsize: Optional[Tuple] = None):
    """
    Creates a Data-Signal-Weights (DSW) plot.
    """
    # TODO: should we drop 0 signal/ weight values?
    index = data.index.intersection(signal.index.intersection(weights.index))
    data = data.reindex(index)
    signal = signal.reindex(index)
    weights = weights.reindex(index)

    if price is not None:
        num_plots = 5
        start_time = price.first_valid_index()
    else:
        num_plots = 3
        start_time = weights.first_valid_index()

    data = data.loc[start_time:]
    signal = signal.loc[start_time:]
    weights = weights.loc[start_time:]

    if figsize is None:
        figsize = (12, num_plots * 1.5)

    fig, axes = plt.subplots(num_plots, figsize=figsize, sharex=True)

    axes[0].text(0.005, 0.83, "Data", fontsize=10, transform=axes[0].transAxes)
    data.plot(ax=axes[0], xlabel="")

    axes[1].text(0.005, 0.83, "Signal", fontsize=10, transform=axes[1].transAxes)
    signal.plot(ax=axes[1], xlabel="")
    axes[1].axhline(0, c="black", ls="--", alpha=0.5)

    axes[2].text(0.005, 0.83, "Weights", fontsize=10, transform=axes[2].transAxes)
    weights.plot(ax=axes[2], xlabel="")
    axes[2].axhline(0, c="black", ls="--", alpha=0.5)

    if price is not None:
        price = price.reindex(index)
        returns = price.pct_change(fill_method=None)
        pnl = weights.shift() * returns

        large_return_periods = compute_large_return_periods(
            pnl, window=large_return_window, num_periods=num_large_return_periods,
            large_return_threshold=large_return_threshold
        )

        axes[3].text(0.005, 0.83, "Price", fontsize=10, transform=axes[3].transAxes)
        price.plot(ax=axes[3], xlabel="")

        axes[4].text(0.005, 0.83, "Pnl", fontsize=10, transform=axes[4].transAxes)
        pnl.cumsum().dropna().plot(ax=axes[4], xlabel="")
        axes[4].axhline(0, c="black", ls="--", alpha=0.5)

        for i in range(num_plots):
            for j in range(len(large_return_periods)):
                return_period = large_return_periods.iloc[j]

                axes[i].text(return_period["start"], 0.5, j + 1,
                             transform=blended_transform_factory(
                                 axes[i].transData, axes[i].transAxes))
                axes[i].axvspan(
                    return_period["start"],
                    return_period["end"],
                    color="grey",
                    alpha=0.3,
                )
    plt.tight_layout()


def plot_df(data: pd.DataFrame, ncols: int = 3, width: float = 20,
            row_height: float = 2, c: str = "#1f77b4"):
    columns = data.columns
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(width, row_height * nrows))
    axes = np.ravel(axes)
    for i, col in enumerate(columns):
        axes[i].plot(data[col], c=c)
        axes[i].set_title(col)
    plt.tight_layout()


def plot_series(data: pd.Series, ncols: int = 3, width: float = 20,
                row_height: float = 2, c: str = "#1f77b4", marker: Optional[str] = "o"):
    values = sorted(data.index.unique(level=1))
    nrows = math.ceil(len(values) / 3)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(width, row_height * nrows))
    axes = np.ravel(axes)
    for i, value in enumerate(values):
        axes[i].plot(data.xs(value, level=1), c=c, marker=marker)
        axes[i].set_title(value)
    plt.tight_layout()


def color_fn(v: float, max_abs_value: float, pos_color: str = "darkgreen",
             neg_color: str = "darkred") -> str:
    if max_abs_value == 0:
        raise ValueError("max_abs_value must be greater than zero.")

    if v == 0 or np.isnan(v):
        return "#00000000"

    v_clamped = max(-max_abs_value, min(max_abs_value, v))
    intensity = abs(v_clamped) / max_abs_value
    if v > 0:
        color = mcolors.to_rgba(pos_color, alpha=intensity)
    else:
        color = mcolors.to_rgba(neg_color, alpha=intensity)
    return mcolors.to_hex(color, keep_alpha=True)


def plot_events(events: pd.Series, ncols: int = 3, bar_width: float = 1,
                sharex: bool = True):
    events = events.replace([np.inf, -np.inf], np.nan).dropna()
    keys = sorted(list(events.index.unique(level=1)))
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(20, 2.5 * nrows),
        sharex=sharex,
    )
    axes = np.ravel(axes)
    for i, key in enumerate(keys):
        events_ = events.xs(key, level=1)
        axes[i].set_title(key)
        if not events_.empty:
            max_abs_value = events_.abs().max()
            colors = [color_fn(x, max_abs_value) for x in events_.values]
            axes[i].bar(
                events_.index, events_.values, color=colors, width=bar_width, alpha=0.5
            )
    plt.tight_layout()
