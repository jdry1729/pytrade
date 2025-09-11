from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from pytrade.stats.lm import compute_t_and_p_values
from pytrade.utils.collections import is_iterable_of
from pytrade.utils.pandas import pandas_to_numpy
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


@dataclass
class _NumpySinglePeriodFactorLoadingModel:
    factor_loadings: np.ndarray
    pvalues: np.ndarray
    r2: np.ndarray
    sample_size: np.ndarray


@dataclass
class _NumpyFactorLoadingModel:
    factor_loadings: np.ndarray
    pvalues: np.ndarray
    r2: np.ndarray
    sample_size: np.ndarray


@dataclass
class FactorLoadingModel:
    factor_loadings: pd.DataFrame
    pvalues: pd.DataFrame
    r2: pd.DataFrame
    sample_size: pd.DataFrame


def _numpy_fit_single_period_factor_loading_model(
        returns: np.ndarray,
        factor_returns: np.ndarray,
        min_sample_size: Optional[int] = 5,
        min_nonzero_factor_returns: Optional[int] = None,
) -> _NumpySinglePeriodFactorLoadingModel:
    if min_sample_size is None:
        min_sample_size = 1

    T, N = returns.shape
    K = factor_returns.shape[1]

    r2 = np.nan
    factor_loadings = np.full((K, N), np.nan)
    pvalues = np.full((K, N), np.nan)
    r2s = np.full(N, np.nan)
    sample_sizes = np.full(N, np.nan)

    factor_mask = (
            np.count_nonzero(factor_returns, axis=0) < min_nonzero_factor_returns
    )
    factor_returns = factor_returns[:, ~factor_mask]

    for i in range(N):
        returns_ = returns[:, i]

        # remove samples with nan in
        sample_mask = np.isnan(returns_)
        sample_mask = sample_mask | np.any(np.isnan(factor_returns), axis=1)

        if np.sum(~sample_mask) > min_sample_size:
            returns_ = returns_[~sample_mask]
            factor_returns_ = factor_returns[~sample_mask, :]

            model = LinearRegression(fit_intercept=False)
            model = model.fit(factor_returns_, returns_)
            r2 = model.score(factor_returns_, returns_)
            _, pvalues_ = compute_t_and_p_values(
                factor_returns_, returns_, model.coef_
            )

            factor_loadings[~factor_mask, i] = model.coef_
            pvalues[~factor_mask, i] = pvalues_
            r2s[i] = r2
            sample_sizes[i] = len(returns_)

    return _NumpySinglePeriodFactorLoadingModel(
        factor_loadings=factor_loadings,
        pvalues=pvalues,
        r2=r2,
        sample_size=sample_sizes,
    )


def _numpy_fit_factor_loading_model(
        returns: np.ndarray,
        factor_returns: np.ndarray,
        window_size: int = 60,
        min_sample_size: Optional[int] = 5,
        min_nonzero_factor_returns: Optional[int] = None,
) -> _NumpyFactorLoadingModel:
    T, N = returns.shape
    K = factor_returns.shape[1]

    factor_loadings = np.full((T, K, N), np.nan)
    pvalues = np.full((T, K, N), np.nan)
    r2s = np.full((T, N), np.nan)
    sample_sizes = np.full((T, N), np.nan)

    for i in tqdm(range(window_size - 1, T)):
        returns_slice = returns[i + 1 - window_size: i + 1]
        factor_returns_slice = factor_returns[i + 1 - window_size: i + 1]
        mod = _numpy_fit_single_period_factor_loading_model(
            returns_slice,
            factor_returns_slice,
            min_sample_size=min_sample_size,
            min_nonzero_factor_returns=min_nonzero_factor_returns,
        )
        factor_loadings[i, :, :] = mod.factor_loadings
        pvalues[i, :, :] = mod.pvalues
        r2s[i, :] = mod.r2
        sample_sizes[i, :] = mod.sample_size

    return _NumpySinglePeriodFactorLoadingModel(
        factor_loadings=factor_loadings,
        pvalues=pvalues,
        r2=r2s,
        sample_size=sample_sizes,
    )


def _pandas_fit_factor_loading_model(
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        *,
        window_size: int = 60,
        min_sample_size: Optional[int] = 5,
        min_nonzero_factor_returns: int = 1
) -> FactorLoadingModel:
    times = returns.index
    assets = returns.columns
    factors = factor_returns.columns

    mod = _numpy_fit_factor_loading_model(
        pandas_to_numpy(returns),
        pandas_to_numpy(factor_returns),
        window_size=window_size,
        min_sample_size=min_sample_size,
        min_nonzero_factor_returns=min_nonzero_factor_returns,
    )

    factor_loadings = pd.DataFrame(
        np.row_stack(mod.factor_loadings),
        index=pd.MultiIndex.from_product([times, factors]),
        columns=assets,
    )
    pvalues = pd.DataFrame(
        np.row_stack(mod.pvalues),
        index=pd.MultiIndex.from_product([times, factors]),
        columns=assets,
    )
    r2 = pd.DataFrame(mod.r2, index=times, columns=assets)
    sample_sizes = pd.DataFrame(mod.sample_size, index=times, columns=assets)
    return FactorLoadingModel(
        factor_loadings=factor_loadings,
        pvalues=pvalues,
        r2=r2,
        sample_size=sample_sizes,
    )


def fit_factor_loading_model(returns, factor_returns,
                             window_size: int = 60,
                             min_sample_size: Optional[int] = 5,
                             min_nonzero_factor_returns: int = 1):
    array_like = [returns, factor_returns]
    if is_iterable_of(array_like, pd.DataFrame):
        return _pandas_fit_factor_loading_model(
            returns, factor_returns,
            window_size=window_size,
            min_sample_size=min_sample_size,
            min_nonzero_factor_returns=min_nonzero_factor_returns)
    elif is_iterable_of(array_like, np.ndarray):
        return _numpy_fit_factor_loading_model(
            returns, factor_returns,
            window_size=window_size,
            min_sample_size=min_sample_size,
            min_nonzero_factor_returns=min_nonzero_factor_returns)
    raise ValueError(
        "returns and factor returns must either both be dataframes"
        " or all be numpy arrays")
