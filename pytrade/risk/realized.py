from typing import Optional, Union, List

import numpy as np
import pandas as pd
from pytrade.portfolio.analysis import compute_portfolio_returns
from pytrade.risk.models.cov import compute_factor_beta, compute_factor_corr
from pytrade.utils.collections import is_iterable_of
from pytrade.utils.pandas import stack


def _pandas_realized_portfolio_cov(
        portfolio_weights: Union[pd.DataFrame, List[pd.DataFrame]],
        asset_returns: pd.DataFrame, lookback: int = 100,
        ann_factor: Optional[int] = None):
    """
    Computes realized portfolio covariance.

    Parameters
    ----------
    portfolio_weights
        Portfolio weights. If single dataframe, must have multi-index of time
        and portfolio.
    asset_returns
        Asset returns.
    lookback
        Lookback to use to compute realized covariance.
    ann_factor
        Optional annualization factor. By default, no annualization is
        performed.

    Returns
    -------
    Realized portfolio covariance.
    """
    if isinstance(portfolio_weights, list):
        portfolio_weights = stack(portfolio_weights)

    returns = compute_portfolio_returns(portfolio_weights, asset_returns)
    # TODO: don't we need groupby below?
    portfolio_cov = returns.rolling(lookback).cov()
    if ann_factor is not None:
        portfolio_cov *= np.sqrt(ann_factor)

    return portfolio_cov


def realized_portfolio_cov(portfolio_weights, asset_returns, lookback=100,
                           ann_factor=None):
    if (isinstance(portfolio_weights, pd.DataFrame) or is_iterable_of(
            portfolio_weights, pd.DataFrame)) and isinstance(asset_returns,
                                                             pd.DataFrame):
        return _pandas_realized_portfolio_cov(portfolio_weights, asset_returns,
                                              lookback, ann_factor)
    # TODO: numpy implementation
    raise ValueError(
        "portfolio_weights and asset_returns must both be dataframes")


def _pandas_realized_portfolio_vol(portfolio_weights, asset_returns,
                                   lookback=100, min_periods=90,
                                   ann_factor=None):
    returns = (portfolio_weights.shift(1) * asset_returns).sum(axis=1)
    portfolio_vol = returns.rolling(lookback, min_periods=min_periods).std()
    if ann_factor is not None:
        portfolio_vol *= np.sqrt(ann_factor)

    return portfolio_vol


def realized_portfolio_vol(portfolio_weights, asset_returns, lookback=100,
                           min_periods=90, ann_factor=None):
    if isinstance(portfolio_weights, pd.DataFrame) and \
            isinstance(asset_returns, pd.DataFrame):
        return _pandas_realized_portfolio_vol(portfolio_weights, asset_returns,
                                              lookback, min_periods, ann_factor)
    # TODO: numpy implementation
    raise ValueError(
        "portfolio_weights and asset_returns must both be dataframes")


def compute_realized_cov(returns: pd.DataFrame, freq: str = "1D",
                         min_sample_size: int = 5):
    """
    Computes realized covariance matrix.

    Parameters
    ----------
    returns
        High-frequency returns. If any NaN returns exist for an asset for a
        particular period, the realized covariances corresponding to that asset
        will all be NaN.
    freq
        Frequency to compute realized covariance matrix for.
    min_sample_size
        Minimum number of observations required to estimate realized covariance.

    Returns
    -------
    Realized covariance matrix.
    """
    # must set closed right since, for example, the 5-minute return at
    # 2024-02-14 00:00, which corresponds to the return for the period from
    # 2024-02-13 23:55 to 2024-02-14 00:00, should be used to compute the realized
    # covariance for the 2024-02-13
    # also set label to right because realized vol for day T is only known at
    # start of day T + 1
    grouper = pd.Grouper(freq=freq, closed="right", label="right")
    group_size = returns.groupby(grouper).size()
    periods = group_size.max()
    valid_times = group_size[group_size > 0].index

    def compute_cov(returns_: pd.DataFrame):
        assets = returns_.columns

        returns_ = returns_.values
        mask = (~np.isnan(returns_)).astype(int)
        sample_size_ = mask.T @ mask

        returns_ = np.nan_to_num(returns_)
        # must multiply by periods to scale to freq
        cov = periods / (sample_size_ - 1) * (returns_.T @ returns_)
        cov = np.where(sample_size_ >= min_sample_size, cov, np.nan)
        return pd.DataFrame(cov, index=assets, columns=assets)

    cov = returns.groupby(grouper).apply(compute_cov)
    cov = cov.loc[cov.index.get_level_values(0).isin(valid_times)]
    return cov.replace([-np.inf, np.inf], np.nan)


def compute_realized_factor_beta(portfolio_weights, factor_portfolio_weights,
                                 realized_asset_cov):
    """
    Computes realized factor beta.

    Notes
    -----
    Since portfolio weights change during the day, this doesn't give the exact same
    value as what you'd get if you computed realized beta from high-frequency returns
    of portfolio and factor portfolios - however it's much simpler to implement and
    is a very good approximation.
    """
    if isinstance(factor_portfolio_weights, dict):
        factor_portfolio_weights = stack(factor_portfolio_weights, names=["factor"])
    # must shift weights so realized beta at time T is estimated over T-1
    return compute_factor_beta(
        portfolio_weights.shift(),
        factor_portfolio_weights.groupby(level=1).shift(),
        realized_asset_cov.fillna(0),
    )


def compute_realized_factor_corr(portfolio_weights: pd.DataFrame,
                                 factor_portfolio_weights: pd.DataFrame,
                                 realized_asset_cov: pd.DataFrame):
    if isinstance(factor_portfolio_weights, dict):
        factor_portfolio_weights = stack(factor_portfolio_weights, names=["factor"])
    # must shift weights so realized beta at time T is estimated over T-1
    return compute_factor_corr(
        portfolio_weights.shift(),
        factor_portfolio_weights.groupby(level=1).shift(),
        realized_asset_cov.fillna(0),
    )
