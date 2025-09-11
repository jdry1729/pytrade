from typing import Optional, Union, Dict, List

import numpy as np
import pandas as pd

from pytrade.signal.masks import EntryCondition, entry_exit_mask
from pytrade.utils.position import Direction


def entry_exit_signal(data: pd.DataFrame, entry_threshold: float,
                      exit_threshold: float,
                      entry_condition: EntryCondition =
                      EntryCondition.GREATER_THAN,
                      trade_direction: Direction = Direction.LONG):
    """
    Computes entry-exit signal.

    Parameters
    ----------
    data
        Data.
    entry_threshold
        Entry threshold.
    exit_threshold
        Exit threshold.
    entry_condition
        Entry condition.
    trade_direction
        Trade direction.

    Returns
    -------
    Signal.
    """
    mask = entry_exit_mask(data, entry_threshold, exit_threshold,
                           entry_condition)
    return mask * trade_direction


def _compute_vmom(data: Union[pd.Series, pd.DataFrame], speed: int, alpha: float = 0.03,
                  min_periods: Optional[int] = None, ignore_na=True,
                  breaks: Optional[List] = None, break_lookback: int = 0):
    res = []

    if breaks is None:
        breaks = []

    min_index = data.index[0]
    max_index = data.index[-1]

    breaks = sorted(breaks)
    breaks = [x for x in breaks if min_index < x < max_index]
    if min_index not in breaks:
        breaks = [min_index] + breaks

    fast = 1.0 / speed
    slow = 1.0 / (3 * speed)

    for i in range(len(breaks)):
        mask = data.index >= breaks[i]
        if i < len(breaks) - 1:
            mask = mask & (data.index < breaks[i + 1])
        data_ = data[mask]
        fast_ewma = data_.ewm(alpha=fast, min_periods=min_periods,
                              ignore_na=ignore_na).mean()
        slow_ewma = data_.ewm(alpha=slow, min_periods=min_periods,
                              ignore_na=ignore_na).mean()
        vol = data_.ewm(alpha=alpha, min_periods=min_periods, ignore_na=ignore_na).std()
        res.append((fast_ewma - slow_ewma) / vol)

    return pd.concat(res)


def compute_vmom(data: pd.DataFrame, speed: int, alpha: float = 0.03,
                 min_periods: Optional[int] = None, ignore_na: bool = True,
                 breaks: Optional[Dict] = None):
    """
    Computes volatility-momentum signal.

    Parameters
    ----------
    data
    speed
    alpha
    min_periods
    ignore_na
    breaks
        Dictionary of breaks to use when computing signal. Must be a map from column
        name to a list of datetimes.

    Returns
    -------
    Signal.
    """
    if breaks is None:
        breaks = {}

    res = []
    for column in data.columns:
        res.append(_compute_vmom(
            data[column], speed=speed, alpha=alpha, min_periods=min_periods,
            ignore_na=ignore_na, breaks=breaks.get(column)).rename(column))
    return pd.concat(res, axis=1)


def compute_macd(data: Union[pd.DataFrame, pd.Series],
                 fast_halflife: float, slow_halflife: float,
                 min_periods: Optional[int] = None,
                 ignore_na: bool = True, log: bool = False,
                 normalize: bool = False) -> pd.DataFrame:
    """
    Computes MACD signal.
    """
    kwargs = {"min_periods": min_periods, "ignore_na": ignore_na}
    invalid_mask = data.isnull()
    fast = data.ewm(halflife=fast_halflife, **kwargs).mean()
    slow = data.ewm(halflife=slow_halflife, **kwargs).mean()
    if log:
        fast = np.log(fast)
        slow = np.log(slow)
    signal = fast - slow
    if normalize:
        signal /= slow
    return signal[~invalid_mask]
