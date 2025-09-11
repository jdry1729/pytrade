import numpy as np
import pandas as pd
from pytrade.utils.linalg import svd
from tqdm import tqdm


def _numpy_compute_single_period_pca_cov(returns: np.ndarray, num_factors: int):
    """
    Computes loadings and specific variance for statistical (i.e., PCA) factor
    model.

    Parameters
    ----------
    returns
        Returns. Must have shape (N x N).
    num_factors
        Number of factors to use, K.

    Returns
    -------
    Loadings : (K x N)
        Loadings to each factor.
    Specific variance : (N,)
        Specific variance of each asset.
    """
    # TODO: optionally demean returns? below assumes returns are demeaned
    # TODO: handle case where asset has lots of missing returns, e.g., by
    #  regressing on factor scores
    T = len(returns)
    K = num_factors

    # letting X denote the matrix of returns, the columns of V give the
    # eigenvectors of X'X and the squared diagonal entries of S give the
    # eigenvalues of X'X; the eigenvectors of the sample covariance matrix
    # X'X/(T-1) are equal to the columns of V, and the eigenvalues are
    # found by squaring the diagonal entries of S and dividing by T-1
    u, s, v = svd(returns)
    loadings = (v[:, :K] @ s[:K, :K] / np.sqrt(T - 1)).T  # K x N
    factor_scores = u[:, :K] * np.sqrt(T - 1)  # T x K
    residuals = returns - factor_scores @ loadings  # T x N
    specific_var = np.var(residuals, axis=0, ddof=1)

    return loadings, specific_var


def _numpy_compute_pca_cov(returns: np.ndarray, num_factors: int,
                           window_size: int = 100):
    T, N = returns.shape
    K = num_factors

    loadings = np.empty((T, K, N))
    loadings[:] = np.nan
    specific_var = np.empty((T, N))
    specific_var[:] = np.nan

    for i in tqdm(range(window_size - 1, T)):
        returns_slice = returns[i + 1 - window_size:i + 1]
        l, s = _numpy_compute_single_period_pca_cov(returns_slice, num_factors)
        loadings[i] = l
        specific_var[i] = s

    return loadings, specific_var


def _pandas_compute_pca_cov(returns: pd.DataFrame, num_factors: int,
                            window_size: int = 100):
    times = returns.index
    assets = returns.columns
    loadings, specific_var = _numpy_compute_pca_cov(returns.values, num_factors,
                                                    window_size)

    loadings = pd.DataFrame(np.row_stack(loadings),
                            index=pd.MultiIndex.from_product(
                                [times, range(num_factors)]),
                            columns=assets)
    specific_var = pd.DataFrame(specific_var, index=times, columns=assets)
    return loadings, specific_var


def compute_pca_cov(returns, num_factors, window_size: int = 100):
    if isinstance(returns, pd.DataFrame):
        return _pandas_compute_pca_cov(returns, num_factors, window_size)
    elif isinstance(returns, np.ndarray):
        return _numpy_compute_pca_cov(returns, num_factors, window_size)
    raise ValueError("returns must be a dataframe or numpy array")


def beta_from_pca_loadings(loadings: pd.DataFrame,
                           ann_factor: int = 252) -> pd.Series:
    """
    Estimates beta using the loadings to the first principal component.

    Parameters
    ----------
    loadings
        Factor loadings.
    ann_factor
        Annualization factor.

    Returns
    -------
    Market beta.
    """
    # TODO: why multiply by ann_factor?
    beta = loadings.xs(0, level=1) * np.sqrt(ann_factor)
    # change sign so beta has mostly positive values
    beta = beta.mul(np.sign(beta.median(axis=1)), axis=0)
    # TODO: clip betas cross-sectionally and impute NaNs
    return beta
