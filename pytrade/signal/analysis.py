import logging
from dataclasses import dataclass
from typing import Optional, Dict, Sequence

import numpy as np
import pandas as pd

from pytrade.data.processing import discretize
from pytrade.stats.lm import lm
from pytrade.utils.pandas import raise_if_index_not_equal, count_nonzero
from pytrade.utils.pandas import stack

logger = logging.getLogger(__name__)


@dataclass
class SignalAnalytics:
    lm_coefs: pd.DataFrame
    xs_corr: pd.DataFrame
    auto_corr: pd.DataFrame
    nonzero_count: pd.Series


def compute_lm_coefs(alpha: pd.DataFrame, asset_returns: pd.DataFrame,
                     groups: Optional[Dict] = None) -> pd.DataFrame:
    """
    Computes coefficents associated with linear regression of returns on forecasts
    of returns.

    Parameters
    ----------
    alpha
        Predictions of future returns. Row at time T corresponds to prediction for
        period from T to T + 1.
    asset_returns
        Asset returns. Row at time T represents return from T - 1 to T.
    groups
        Optional dictionary from group names to lists of assets to include in the
        group.

    Returns
    -------
    Model coefficients, num samples and p-values.
    """
    # TODO: multiple horizons?
    assets = alpha.columns
    if groups is None:
        groups = {"all": assets}

    group_keys = tuple(groups.keys())
    K = len(group_keys)

    if alpha.index.nlevels == 1:
        alpha = stack([alpha], names=["alpha"])

    alphas = alpha
    levels = list(range(1, alphas.index.nlevels))

    raise_if_index_not_equal(alphas.columns, asset_returns.columns)

    res = {}
    for key, alpha in alphas.groupby(level=levels):
        alpha = alpha.droplevel(levels)
        res_ = []
        for i in range(K):
            assets_ = groups[group_keys[i]]
            alpha_ = alpha[assets_].shift(1).values.reshape(-1)
            asset_returns_ = asset_returns[assets_].values.reshape(-1)
            nan_mask = np.isnan(alpha_) | np.isnan(asset_returns_)
            X = np.expand_dims(alpha_[~nan_mask], 1)
            # add constant to regression
            X = np.column_stack([np.ones(len(X)), X])
            y = asset_returns_[~nan_mask]
            coefs, _, p_values = lm(X, y)
            # TODO: also report alpha pvalue?
            for j, coef in enumerate(["alpha", "beta"]):
                res_.append({"coef": coef, "value": coefs[j],
                             "p_value": p_values[j], "n": len(X),
                             "group": group_keys[i]})
        res[key] = pd.DataFrame(res_).set_index(["group", "coef"])

    res = stack(res, names=alphas.index.names[1:])
    if K == 1:
        res = res.droplevel("group")
    return res


def compute_xs_corr(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cross-sectional correlation of signals at each timestep.

    Parameters
    ----------
    signals
        Must have multiindex of time and any number of other levels to identify
        each signal.

    Returns
    -------
    Cross-sectional correlation of signals at each timestep.
    """
    nlevels = signals.index.nlevels
    times = signals.index.unique(level=0)

    res = {}
    for time in times:
        res[time] = signals.xs(time, level=0).T.corr()
    res = stack(res, names=["time"])
    return res.reorder_levels((-1, *list(range(nlevels - 1)))).sort_index()


def compute_auto_corr(signal: pd.DataFrame, lags=(1, 5, 10)) -> pd.DataFrame:
    if signal.index.nlevels == 1:
        signal = stack([signal], names=["signal"])

    levels = list(range(1, signal.index.nlevels))

    res = {}
    for lag in lags:
        res[lag] = signal.corrwith(signal.groupby(level=levels).shift(lag), axis=1)
    res = stack(res, names=["lag"])
    # TODO: handle squeeze=True
    return res


def compute_quantile_returns(asset_returns, signal, q: int = 5):
    signal = discretize(signal.shift(), q)
    quantile_returns = []
    for i in range(q):
        quantile_returns.append(asset_returns[signal == i].mean(axis=1))
    return pd.concat(quantile_returns, keys=range(q), axis=1)


def analyse_signal(signal: pd.DataFrame, asset_returns: pd.DataFrame,
                   asset_vol: pd.DataFrame, groups: Optional[Dict] = None,
                   lags: Sequence[int] = (1, 2, 5)) -> SignalAnalytics:
    # assume level 0 is time below
    alpha = signal.mul(asset_vol, level=0)
    lm_coefs = compute_lm_coefs(alpha, asset_returns, groups)
    xs_corr = compute_xs_corr(signal)
    auto_corr = compute_auto_corr(signal, lags)
    nonzero_count = count_nonzero(signal)
    return SignalAnalytics(
        lm_coefs=lm_coefs,
        xs_corr=xs_corr,
        auto_corr=auto_corr,
        nonzero_count=nonzero_count,
    )
