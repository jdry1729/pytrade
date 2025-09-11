from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from pytrade.stats.lm import compute_t_and_p_values
from pytrade.utils.collections import is_iterable_of
from pytrade.utils.numpy import shift
from pytrade.utils.pandas import pandas_to_numpy


@dataclass
class _NumpySinglePeriodFactorReturnModel:
    factor_returns: np.ndarray
    specific_returns: np.ndarray
    pvalues: np.ndarray
    r2: float
    adj_r2: float
    sample_size: int


@dataclass
class _NumpyFactorReturnModel:
    factor_returns: np.ndarray
    specific_returns: np.ndarray
    pvalues: np.ndarray
    r2: np.ndarray
    adj_r2: np.ndarray
    sample_size: np.ndarray


@dataclass
class FactorReturnModel:
    factor_returns: pd.DataFrame
    specific_returns: pd.DataFrame
    pvalues: pd.DataFrame
    r2: pd.Series
    adj_r2: pd.Series
    sample_size: pd.Series


def _numpy_fit_single_period_factor_return_model(
        returns: np.ndarray,
        loadings: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        min_nonzero_loadings: int = 1,
) -> _NumpySinglePeriodFactorReturnModel:
    """
    Fits single period factor return model.

    Parameters
    ----------
    returns
        Array of shape (N,).
    loadings
        Array of shape (K, N).
    weights
        Optional array of shape (N,).
    min_nonzero_loadings

    Returns
    -------
    Estimated factor returns.
    """
    # TODO: if weight is nan for a sample, we still want to estimate specific
    #  return for it! can we just do nan_to_num on weights?
    K, N = loadings.shape
    loadings = loadings.T

    if weights is None:
        weights = np.full(N, 1)

    sample_size = 0
    r2 = np.nan
    adj_r2 = np.nan
    factor_returns = np.full(K, np.nan)
    specific_returns = np.full(N, np.nan)
    pvalues = np.full(K, np.nan)

    # remove samples with nan in
    sample_mask = np.isnan(returns)
    sample_mask = sample_mask | np.any(np.isnan(loadings), axis=1)
    sample_mask = sample_mask | np.isnan(weights)

    returns = returns[~sample_mask]
    loadings = loadings[~sample_mask, :]
    weights = weights[~sample_mask]

    # remove null factors
    factor_mask = np.count_nonzero(loadings, axis=0) < min_nonzero_loadings
    loadings = loadings[:, ~factor_mask]

    N_, K_ = loadings.shape
    if not np.all(factor_mask) and np.abs(np.sum(weights)) > 0:
        model = LinearRegression(fit_intercept=False)
        model = model.fit(loadings, returns, sample_weight=weights)
        preds = model.predict(loadings)
        r2 = model.score(loadings, returns, sample_weight=weights)
        adj_r2 = 1 - ((1 - r2) * (N_ - 1)) / (N_ - K_ - 1)
        _, pvalues_ = compute_t_and_p_values(loadings, returns, model.coef_,
                                             weights=weights)
        sample_size = len(returns)

        factor_returns[~factor_mask] = model.coef_
        specific_returns[~sample_mask] = returns - preds
        pvalues[~factor_mask] = pvalues_

    return _NumpySinglePeriodFactorReturnModel(
        factor_returns=factor_returns, specific_returns=specific_returns,
        pvalues=pvalues, r2=r2, adj_r2=adj_r2, sample_size=sample_size
    )


def _numpy_fit_factor_return_model(
        returns: np.ndarray,
        loadings: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        min_nonzero_loadings: int = 1) -> _NumpyFactorReturnModel:
    """
    Fits factor return model.

    Parameters
    ----------
    returns
        Returns. Return at time T gives the return from T-1 to T.
    loadings
        Loadings. Loadings for time T must be known at time T.
    weights
    min_nonzero_loadings

    Returns
    -------
    Estimated factor and specific returns.

    Notes
    -----
    To estimate the factor returns at time T we regress the returns for time T on
    the loadings corresponding to T-1.
    """
    factor_returns = []
    specific_returns = []
    pvalues = []
    r2s = []
    adj_r2s = []
    sample_sizes = []
    T = returns.shape[0]
    # must shift loadings and weights forward!
    loadings = shift(loadings, 1)
    weights = shift(weights, 1)
    for i in range(T):
        weights_ = None if weights is None else weights[i]
        mod = _numpy_fit_single_period_factor_return_model(
            returns[i], loadings[i], weights=weights_,
            min_nonzero_loadings=min_nonzero_loadings)
        factor_returns.append(mod.factor_returns)
        specific_returns.append(mod.specific_returns)
        pvalues.append(mod.pvalues)
        r2s.append(mod.r2)
        adj_r2s.append(mod.adj_r2)
        sample_sizes.append(mod.sample_size)
    return _NumpyFactorReturnModel(
        factor_returns=np.vstack(factor_returns),
        specific_returns=np.vstack(specific_returns),
        pvalues=np.vstack(pvalues),
        r2=np.array(r2s),
        adj_r2=np.array(adj_r2s),
        sample_size=np.array(sample_sizes),
    )


def _pandas_fit_factor_return_model(
        returns: pd.DataFrame,
        loadings: pd.DataFrame,
        *,
        weights: Optional[pd.DataFrame] = None,
        min_nonzero_loadings: int = 1
) -> FactorReturnModel:
    if weights is not None:
        weights = pandas_to_numpy(weights)
    mod = _numpy_fit_factor_return_model(
        pandas_to_numpy(returns),
        pandas_to_numpy(loadings), weights=weights,
        min_nonzero_loadings=min_nonzero_loadings)
    factors = loadings.index.unique(level=1)
    factor_returns = pd.DataFrame(mod.factor_returns, index=returns.index,
                                  columns=factors)
    specific_returns = pd.DataFrame(mod.specific_returns, index=returns.index,
                                    columns=returns.columns)
    pvalues = pd.DataFrame(mod.pvalues, index=returns.index, columns=factors)
    return FactorReturnModel(factor_returns=factor_returns,
                             specific_returns=specific_returns,
                             pvalues=pvalues,
                             r2=pd.Series(mod.r2, index=returns.index),
                             adj_r2=pd.Series(mod.adj_r2, index=returns.index),
                             sample_size=pd.Series(
                                 mod.sample_size, index=returns.index))


def fit_factor_return_model(returns, loadings, weights=None,
                            min_nonzero_loadings: int = 1):
    array_like = [returns, loadings]
    if is_iterable_of(array_like, pd.DataFrame):
        return _pandas_fit_factor_return_model(
            returns, loadings, weights=weights,
            min_nonzero_loadings=min_nonzero_loadings)
    elif is_iterable_of(array_like, np.ndarray):
        return _numpy_fit_factor_return_model(
            returns, loadings, weights=weights,
            min_nonzero_loadings=min_nonzero_loadings)
    raise ValueError(
        "returns and loadings must either both be dataframes"
        " or all be numpy arrays")
