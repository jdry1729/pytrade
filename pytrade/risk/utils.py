from typing import Union

import numpy as np
import pandas as pd
from pytrade.utils import stack


def scale_vol_by_time(vol: Union[float, pd.Series, pd.DataFrame, np.ndarray],
                      period_ratio: float):
    """

    Parameters
    ----------
    vol
        Volatility.
    period_ratio
        Ratio of period of time to scale to and period to scale from. E.g., if
        you have a 1-day volatility, and want to convert that into an annualized
        volatility, you should set this to the number of trading days in a year.
        To convert the other way, you should set this to the reciprocal of the
        number of trading days in a year.

    Returns
    -------
    Scaled volatility.
    """
    return vol * np.sqrt(period_ratio)


def scale_cov_by_time(cov: Union[float, pd.Series, pd.DataFrame, np.ndarray],
                      period_ratio: float):
    """

    Parameters
    ----------
    cov
        Covariance.
    period_ratio
        Ratio of period of time to scale to and period to scale from. E.g., if
        you have a 1-day covariance, and want to convert that into an annualized
        covariance, you should set this to the number of trading days in a year.
        To convert the other way, you should set this to the reciprocal of the
        number of trading days in a year.
    Returns
    -------
    Scaled covariance.
    """
    return cov * period_ratio


def decompose_returns(factor_returns: pd.DataFrame, specific_returns: pd.DataFrame,
                      loadings: pd.DataFrame) -> pd.DataFrame:
    res = loadings.mul(factor_returns.stack(), axis=0)
    return pd.concat([res, stack([specific_returns], keys=["specific"],
                                 names=["factor"])]).sort_index()
