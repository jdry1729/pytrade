from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_sample_cov(returns: pd.DataFrame, window_size: int = 100,
                       min_periods: int = 90):
    return returns.rolling(window_size, min_periods=min_periods).cov()


def compute_ew_sample_cov(returns: pd.DataFrame, alpha: float = 0.03,
                          min_periods: int = 90, clip: Optional[float] = None):
    if clip is not None:
        returns = returns.clip(-clip, clip)
    return returns.ewm(alpha=alpha, min_periods=min_periods).cov()


def _numpy_compute_single_period_lwcc_cov(returns: np.array):
    """
    Computes Ledoit-Wolf constant correlation covariance matrix.

    Parameters
    ----------
    returns
        Returns.

    Returns
    -------
    Shrunk covaraiance matrix.

    References
    ----------
    Ledoit and Wolf "Honey, I Shrunk the Sample Covariance Matrix", 2004
    """
    T, N = returns.shape

    # TODO: avoid copy!
    returns = returns.copy()
    returns -= np.mean(returns, axis=0)

    sample_cov = 1 / (T - 1) * returns.T @ returns

    # compute target
    sample_var = np.diag(sample_cov)[np.newaxis, :]
    sample_std = np.sqrt(sample_var)
    sample_corr = sample_cov / (sample_std.T @ sample_std)
    np.fill_diagonal(sample_corr, np.nan)
    mean_corr = np.nanmean(sample_corr)
    target = mean_corr * (sample_std.T @ sample_std)
    di = np.diag_indices(N)
    target[di] = sample_cov[di]

    # estimate gamma (target misspecification error)
    gamma = np.nansum((target - sample_cov) ** 2)

    # estimate pi
    returns_sqrd = returns ** 2
    pi_mat = 1 / (T - 1) * returns_sqrd.T @ returns_sqrd - sample_cov ** 2
    pi = np.nansum(pi_mat)

    # estimate rho
    rho_di = np.nansum(np.diag(pi_mat))
    term1 = (returns ** 3).T @ returns / (T - 1)
    term2 = np.tile(sample_var, (N, 1)).T * sample_cov
    theta = term1 - term2
    np.fill_diagonal(theta, np.nan)
    rho_off = mean_corr * np.nansum(((1 / sample_std).T @ sample_std) * theta)
    rho = rho_di + rho_off

    # estimate shrinkage intensity
    kappa = (pi - rho) / gamma
    shrinkage = max(0, min(kappa / T, 1.0))

    return shrinkage * target + (1 - shrinkage) * sample_cov


def _numpy_compute_lwcc_cov(returns: np.array, window_size: int = 10):
    T, N = returns.shape

    cov = np.empty((T, N, N))
    cov[:] = np.nan

    for i in tqdm(range(window_size - 1, T)):
        returns_slice = returns[i + 1 - window_size:i + 1]
        cov[i] = _numpy_compute_single_period_lwcc_cov(returns_slice)

    return cov


def _pandas_compute_lwcc_cov(returns: pd.DataFrame, window_size: int = 100):
    times = returns.index
    assets = returns.columns
    cov = _numpy_compute_lwcc_cov(returns.values, window_size)

    return pd.DataFrame(np.row_stack(cov),
                        index=pd.MultiIndex.from_product([times, assets]),
                        columns=assets)


def compute_lwcc_cov(returns, window_size: int = 100):
    if isinstance(returns, pd.DataFrame):
        return _pandas_compute_lwcc_cov(returns, window_size)
    elif isinstance(returns, np.ndarray):
        return _numpy_compute_lwcc_cov(returns, window_size)
    raise ValueError("returns must either be a dataframe or numpy array")
