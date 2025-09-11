import numpy as np
import pandas as pd
from pytrade.utils import pandas_to_numpy


def _pandas_cov_to_std(cov: pd.DataFrame):
    std = _numpy_cov_to_std(pandas_to_numpy(cov))
    if cov.index.nlevels == 2:
        times = cov.index.unique(0)
        return pd.DataFrame(std, index=times, columns=cov.columns)
    return pd.Series(std, index=cov.columns)


def _numpy_cov_to_std(cov: np.ndarray):
    if cov.ndim == 2:
        var = np.diagonal(cov)
    elif cov.ndim == 3:
        var = np.diagonal(cov, axis1=1, axis2=2)
    else:
        raise ValueError("Error converting covariance to volatility; covariance"
                         " array must be 2 or 3 dimensional")
    return np.sqrt(var)


def cov_to_std(cov):
    if isinstance(cov, pd.DataFrame):
        return _pandas_cov_to_std(cov)
    elif isinstance(cov, np.ndarray):
        return _numpy_cov_to_std(cov)
    raise ValueError("cov must either be a dataframe or numpy array")


def _numpy_cov_to_corr(cov: np.ndarray):
    """
    Converts covariance matrix to cov matrix. Works with 2D and 3D input.
    """
    std = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
    std = np.expand_dims(std, axis=-1) * np.expand_dims(std, axis=-2)
    # TODO: force main diagonal to 1.0?
    return cov / std


def _pandas_cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """
    Converts covariance matrix to cov matrix. Works with 2D and 3D input.

    Parameters
    ----------
    cov
        Covariance matrix. Or many covariance matrices over time.

    Returns
    -------
    Correlation matrix.
    """
    corr = _numpy_cov_to_corr(pandas_to_numpy(cov))
    return pd.DataFrame(np.row_stack(corr), index=cov.index, columns=cov.columns)


def cov_to_corr(cov):
    if isinstance(cov, pd.DataFrame):
        return _pandas_cov_to_corr(cov)
    elif isinstance(cov, np.ndarray):
        return _numpy_cov_to_corr(cov)
    raise ValueError("cov must either be numpy array or dataframe")


def _numpy_zero_non_diag(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        return data * np.eye(data.shape[0])
    elif data.ndim == 3:
        T, N, _ = data.shape
        mask = np.eye(N)[None, :, :]
        return data * mask
    raise ValueError("Array must be 2D or 3D")


def _pandas_zero_non_diag(data: pd.DataFrame) -> pd.DataFrame:
    arr = _numpy_zero_non_diag(pandas_to_numpy(data))
    return pd.DataFrame(np.row_stack(arr), index=data.index, columns=data.columns)


def zero_non_diag(data):
    if isinstance(data, pd.DataFrame):
        return _pandas_zero_non_diag(data)
    elif isinstance(data, np.ndarray):
        return _numpy_zero_non_diag(data)
    raise ValueError("data must either be numpy array or dataframe")
