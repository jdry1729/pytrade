from typing import Optional, Dict
from typing import Union, Iterable, Tuple

import numpy as np
import pandas as pd
from pytrade.risk.models.cov import compute_portfolio_vol, compute_portfolio_cov
from pytrade.utils import stack
from pytrade.utils.collections import is_iterable_of
from pytrade.utils.pandas import pandas_to_numpy, round_
from pytrade.utils.typing import T1


def compute_quantile_portfolios(factor: pd.DataFrame, q: Optional[int] = 5):
    """
    Computes quantile portfolios.

    Parameters
    ----------
    factor
        Factor.
    q
        Number of quantiles.

    Returns
    -------
    Portfolios.
    """
    # TODO: what if factor has multi-index
    # TODO: add option to assign weights in each quantile based on factor score
    portfolios = []
    bins = factor.apply(pd.qcut, axis=1, q=q, labels=False)
    for i in range(q):
        mask = bins == i
        portfolios.append(mask.div(mask.sum(axis=1), axis=0))
    return stack(portfolios)


# TODO: write tests!
def _numpy_risk_orthogonalize(portfolio_weights: np.ndarray,
                              factor_portfolio_weights: np.ndarray,
                              asset_cov: Union[np.ndarray, Tuple[
                                  np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Risk-orthogonalizes portfolio to a number of factor.

    Parameters
    ----------
    portfolio_weights : (T, N)
        Portfolio weights.
    factor_portfolio_weights : (T, M, N)
        Factor portfolio weights.
    asset_cov
        Either a full asset cov matrix with shape (T, N, N), or a tuple of form
        (loadings, factor_cov, specific_var). Loadings must have shape (T, K, N).
        Factor covariance must have shape (T, K, K) and specific var must have
        shape (T, N).

    Returns
    -------
    Orthogonalized portfolio weights.
    """
    M = factor_portfolio_weights.shape[1]

    portfolio_weights = np.expand_dims(portfolio_weights, axis=1)
    portfolio_weights = np.concatenate(
        [factor_portfolio_weights, portfolio_weights], axis=1)

    # compute error when  associated with projection of portfolio weights onto
    # factor portfolio subspace
    for i in range(1, M + 1):
        portfolio_cov = compute_portfolio_cov(portfolio_weights[:, :i + 1], asset_cov)
        portfolio_var = np.diagonal(portfolio_cov, axis1=1, axis2=2)
        # TODO: what if portfolio_var very small?
        coeffs = np.expand_dims(portfolio_cov[:, -1] / portfolio_var, 2)[:, :i]
        # coefficients may be nan if factor portfolio returns aren't linearly
        # independent, so must use nansum below
        portfolio_weights[:, i] -= np.nansum(portfolio_weights[:, :i] * coeffs,
                                             axis=1)
    return portfolio_weights[:, -1]


def _pandas_risk_orthogonalize(portfolio_weights: pd.DataFrame,
                               factor_portfolio_weights: Union[
                                   Iterable[pd.DataFrame], pd.DataFrame],
                               asset_cov: Union[pd.DataFrame, Tuple[
                                   pd.DataFrame, pd.DataFrame, pd.DataFrame]]):
    if isinstance(factor_portfolio_weights, list):
        factor_portfolio_weights = stack(factor_portfolio_weights, names="portfolio")

    orth_weights = _numpy_risk_orthogonalize(
        pandas_to_numpy(portfolio_weights), pandas_to_numpy(factor_portfolio_weights),
        pandas_to_numpy(asset_cov))
    return pd.DataFrame(orth_weights, index=portfolio_weights.index,
                        columns=portfolio_weights.columns)


def risk_orthogonalize(portfolio_weights, factor_portfolio_weights, asset_cov):
    """
    Risk-orthogonalizes portfolio to a number of factor.

    Parameters
    ----------
    portfolio_weights
        Portfolio weights.
    factor_portfolio_weights
        Factor portfolio weights.
    asset_cov
        asset_cov

    Returns
    -------
    Orthogonalized portfolio weights.
    """
    if isinstance(portfolio_weights, pd.DataFrame) and isinstance(
            factor_portfolio_weights, (Iterable, pd.DataFrame)):
        return _pandas_risk_orthogonalize(portfolio_weights,
                                          factor_portfolio_weights,
                                          asset_cov)
    elif is_iterable_of([portfolio_weights, factor_portfolio_weights],
                        np.ndarray):
        return _numpy_risk_orthogonalize(portfolio_weights,
                                         factor_portfolio_weights,
                                         asset_cov)
    raise ValueError(
        "portfolio_weights and factor_portfolio_weights must either both be "
        "dataframes or both be numpy arrays")


def _numpy_vol_scale(portfolio_weights: np.ndarray,
                     asset_cov: Union[np.ndarray, Tuple[
                         np.ndarray, np.ndarray, np.ndarray]],
                     target_vol: Union[float, np.ndarray]):
    """
    Volatility-scales weights.

    Parameters
    ----------
    portfolio_weights
        Portfolio weights. May be of shape (T, N) or (T, K, N) where K is the
        number of portfolios, and N is number of assets.
    asset_cov
        Either a full asset cov matrix with shape (T, N, N), or a tuple of form
        (loadings, factor_cov, specific_var). Loadings must have shape (T, K, N).
        Factor covariance must have shape (T, K, K) and specific var must have
        shape (T, N).
    target_vol
        Target volatility. If portfolio weights is 2-dimensional, may be a float or
        an array of shape (T,). If portfolio weights is 3-dimensional, may be a float
        or an array of either shape (T,) or (T, K).

    Returns
    -------
    out:
        Scaled portfolio weights. Will have same shape as portfolio weights.

    Notes
    -----
    If a full asset covariance matrix is provided, assets with nan variance are
    considered invalid. If a structural risk model is provided, assets with nan
    specific variance or any nan loadings are considered invalid. Invalid assets
    will have a weight of nan in the volatility-scaled portfolio.
    """
    portfolio_weights = portfolio_weights.copy()
    if isinstance(asset_cov, np.ndarray):
        invalid_assets = np.isnan(np.diagonal(asset_cov, axis1=1, axis2=2))
    else:
        invalid_assets = np.any(np.isnan(asset_cov[0]), axis=1)
        invalid_assets = invalid_assets | np.isnan(asset_cov[2])

    if portfolio_weights.ndim == 3:
        invalid_assets = np.broadcast_to(invalid_assets[:, np.newaxis, :],
                                         portfolio_weights.shape)
        if isinstance(target_vol, np.ndarray) and target_vol.ndim == 1:
            target_vol = np.expand_dims(target_vol, -1)

    portfolio_weights[invalid_assets] = np.nan
    portfolio_vol = compute_portfolio_vol(portfolio_weights, asset_cov)
    portfolio_vol = np.expand_dims(portfolio_vol, -1)

    scaling_factor = np.expand_dims(target_vol, -1) / portfolio_vol
    return scaling_factor * portfolio_weights


def _pandas_vol_scale(portfolio_weights: pd.DataFrame,
                      asset_cov: Union[pd.DataFrame, Tuple[
                          pd.DataFrame, pd.DataFrame, pd.DataFrame]],
                      target_vol: Union[float, pd.Series]):
    """
    Volatility-scales weights.
    """
    portfolio_weights_arr = _numpy_vol_scale(pandas_to_numpy(portfolio_weights),
                                             pandas_to_numpy(asset_cov),
                                             pandas_to_numpy(target_vol))
    return pd.DataFrame(np.row_stack(portfolio_weights_arr),
                        index=portfolio_weights.index,
                        columns=portfolio_weights.columns)


def vol_scale(portfolio_weights, asset_cov, target_vol):
    if isinstance(portfolio_weights, pd.DataFrame):
        return _pandas_vol_scale(portfolio_weights, asset_cov, target_vol)
    elif isinstance(portfolio_weights, np.ndarray):
        return _numpy_vol_scale(portfolio_weights, asset_cov, target_vol)
    raise ValueError(
        "portfolio_weights must either be a dataframe of numpy array")


def allocate(data: pd.DataFrame, weights: Union[pd.DataFrame, pd.Series, Dict],
             combine: bool = True) -> pd.DataFrame:
    """
    Computes a weighted average of data.

    Parameters
    ----------
    data
        Data to average. Must have a multiindex of time and one other level.
    weights
        Weights to use for averaging.
    combine
        Whether to aggregate contributions.

    Returns
    -------
    Weighted average.

    Notes
    -----
    If data is nan for a particular combination of index and column for every key,
    the returned average will be nan too.
    """
    if isinstance(data, dict):
        data = stack(data)

    invalid_mask = data.isnull().groupby(level=0).all()

    res = {}
    for k in data.index.unique(level=1):
        res[k] = data.xs(k, level=1).mul(weights[k], axis=0)

    res = stack(res)
    if combine:
        res = res.groupby(level=0).sum()

    return res[~invalid_mask]


def _ignore_small_trades_numpy(positions: np.ndarray, asset_prices: np.ndarray,
                               min_trade_size: float) -> np.ndarray:
    """
    Adjusts positions.

    Parameters
    ----------
    positions
        Target positions.
    asset_prices
        Prices.
    min_trade_size

    Returns
    -------
    Adjusted positions.
    """
    T = positions.shape[0]
    positions = np.nan_to_num(positions)
    positions_adj = np.zeros_like(positions)
    # trades are unknown at time 0, so positions are set to target positions
    positions_adj[0] = positions[0]
    for i in range(1, T):
        init_positions = positions_adj[i - 1]
        target_positions = positions[i]
        trades = target_positions - init_positions
        trades_notional = trades * asset_prices[i]
        trades = np.where(np.abs(trades_notional) >= min_trade_size, trades, 0)
        positions_adj[i] = init_positions + trades
    return positions_adj


def _ignore_small_trades_pandas(positions: pd.DataFrame, asset_prices: pd.DataFrame,
                                min_trade_size: float) -> pd.DataFrame:
    positions_adj = _ignore_small_trades_numpy(
        pandas_to_numpy(positions), pandas_to_numpy(asset_prices),
        min_trade_size)
    return pd.DataFrame(np.row_stack(positions_adj), index=positions.index,
                        columns=positions.columns)


def ignore_small_trades(positions, asset_prices, min_trade_size: float):
    if isinstance(positions, pd.DataFrame):
        return _ignore_small_trades_pandas(positions, asset_prices, min_trade_size)
    elif isinstance(positions, np.ndarray):
        return _ignore_small_trades_numpy(positions, asset_prices, min_trade_size)
    raise ValueError("Error adjusting positions; positions must either be a dataframe"
                     " or array")


def weights_to_positions(portfolio_weights: T1, asset_prices: T1,
                         fum: float, decimals: Optional[int] = None) -> T1:
    positions = (portfolio_weights * fum).div(asset_prices, level=0)
    if decimals is not None:
        positions = round_(positions, decimals)
    return positions


def positions_to_weights(positions: T1, asset_prices: T1, fum: float) -> T1:
    return positions.mul(asset_prices, level=0) / fum
