from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union, Dict, Iterable, Tuple, List, Sequence, Collection

import numpy as np
import pandas as pd

from pytrade.data.mapping import map_ids
from pytrade.utils.collections import is_iterable_of
from pytrade.utils.linalg import compute_orth_basis
from pytrade.utils.pandas import pandas_to_numpy, stack


# TODO: possible to use weighted_avg here? maybe if we stack data first?
def compute_xs_avg(data: pd.DataFrame,
                   weights: Optional[
                       Union[Dict[str, float], pd.DataFrame]] = None):
    """
    Computes cross-sectional weighted average. Weights may change over time.

    Parameters
    ----------
    data
        Data.
    weights
        Weights.
    """
    if weights is None:
        weights = pd.DataFrame(1, index=data.index, columns=data.columns)
    elif isinstance(weights, dict):
        weights = pd.DataFrame(weights, index=data.index)
    # TODO: better way to compute weighted average?
    weights = weights.fillna(0).div(weights.sum(axis=1), axis=0)
    return data.mul(weights).sum(axis=1)


def remove_outliers(data: pd.DataFrame, clip=3, remove=6):
    """
    Removes cross-sectional outliers in the data.
    """
    ranks = data.rank(axis=1, pct=True)
    data_robust = data[(ranks <= 0.95) & (ranks >= 0.05)]
    # TODO: create xs zscore function
    mean = data_robust.mean(axis=1)
    std = data_robust.std(axis=1)
    zscores = data.subtract(mean, axis=0).divide(std, axis=0)
    zscores = zscores.clip(-clip, clip)
    zscores[zscores.abs() > remove] = np.nan
    return zscores.multiply(std, axis=0).add(mean, axis=0)


@dataclass
class ZScoreSpikeDetectionResults:
    valid_mask: Union[np.ndarray, pd.Series]
    spike_mask: Union[np.ndarray, pd.Series]
    zscores: Union[np.ndarray, pd.Series]
    window_means: Union[np.ndarray, pd.Series]
    window_stds: Union[np.ndarray, pd.Series]


def numpy_zscore_detect_spikes(
        data, window_size: int, entry_threshold: float = 6,
        exit_threshold: float = 3,
        max_spike_periods: Optional[int] = None,
        min_spike_periods: Optional[int] = None,
        clear_window_if_max_spike_periods_exceeded: bool = True) \
        -> ZScoreSpikeDetectionResults:
    if min_spike_periods is None:
        min_spike_periods = 1

    spike_periods = 0
    valid_mask = np.full(len(data), False)
    peak_mask = np.full(len(data), False)
    zscores = np.full(len(data), np.nan)
    window_means = np.full(len(data), np.nan)
    window_stds = np.full(len(data), np.nan)

    window = deque(maxlen=window_size)
    for i in range(len(data)):
        if np.isnan(data[i]):
            continue

        if max_spike_periods is not None and spike_periods >= max_spike_periods:
            spike_periods = 0
            if clear_window_if_max_spike_periods_exceeded:
                window.clear()

        if len(window) < window_size:
            window.append(data[i])
            continue

        valid_mask[i] = True

        # noinspection PyTypeChecker
        window_mean = np.nanmean(window)
        window_means[i] = window_mean
        # noinspection PyTypeChecker
        window_std = np.nanstd(window)
        window_stds[i] = window_std

        zscore = (data[i] - window_mean) / window_std
        zscores[i] = zscore
        if spike_periods == 0:
            if abs(zscore) > entry_threshold:
                peak_mask[i] = True
                spike_periods = 1
            else:
                window.append(data[i])
        else:
            if abs(zscore) <= exit_threshold and spike_periods >= min_spike_periods:
                spike_periods = 0
                window.append(data[i])
            else:
                peak_mask[i] = True
                spike_periods += 1

    return ZScoreSpikeDetectionResults(valid_mask=valid_mask,
                                       spike_mask=peak_mask,
                                       zscores=zscores,
                                       window_means=window_means,
                                       window_stds=window_stds)


def zscore_detect_spikes(data, window_size: int, entry_threshold: float = 6,
                         exit_threshold: float = 3,
                         max_spike_periods: Optional[int] = None,
                         min_spike_periods: Optional[int] = None,
                         clear_window_if_max_spike_periods_exceeded: bool = True) -> \
        ZScoreSpikeDetectionResults:
    if isinstance(data, pd.Series):
        results = numpy_zscore_detect_spikes(
            data.values, window_size,
            entry_threshold,
            exit_threshold,
            max_spike_periods,
            min_spike_periods,
            clear_window_if_max_spike_periods_exceeded)
        return ZScoreSpikeDetectionResults(
            valid_mask=pd.Series(results.valid_mask, index=data.index),
            spike_mask=pd.Series(results.spike_mask, index=data.index),
            zscores=pd.Series(results.zscores, index=data.index),
            window_means=pd.Series(results.window_means, index=data.index),
            window_stds=pd.Series(results.window_stds, index=data.index)
        )
    return numpy_zscore_detect_spikes(
        data, window_size, entry_threshold,
        exit_threshold, max_spike_periods,
        min_spike_periods,
        clear_window_if_max_spike_periods_exceeded)


def zscore_remove_spikes(data: pd.Series, window_size: int, entry_threshold: float = 6,
                         exit_threshold: float = 2,
                         max_spike_periods: Optional[int] = None,
                         min_spike_periods: Optional[int] = None,
                         clear_window_if_max_spike_periods_exceeded: bool = True,
                         clip_threshold: Optional[float] = None):
    results = zscore_detect_spikes(data, window_size, entry_threshold,
                                   exit_threshold, max_spike_periods,
                                   min_spike_periods,
                                   clear_window_if_max_spike_periods_exceeded)
    clean_data = data.mask((~results.valid_mask) | results.spike_mask)
    if clip_threshold is not None:
        clip_values = (clip_threshold * np.sign(results.zscores) * results.window_stds
                       + results.window_means)
        clean_data = clean_data.fillna(clip_values)
    return clean_data


def _numpy_orth(data: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """
    Orthogonalizes data by subtracting its projection onto the subspace spanned by
    the factor. The result is equivalent to the residuals of the regression of data
    on factors at each timestep.

    Parameters
    ----------
    data : (T, N)
        Time-series data.
    factors : (T, K, N)
        Factors.

    Returns
    -------
    out : (T, N)
        Orthogonalized data.
    """
    T, N = data.shape
    K = factors.shape[1]
    factors = factors.transpose(0, 2, 1)

    bases = []
    invalid_mask = np.isnan(data) | np.any(np.isnan(factors), axis=2)
    for i in range(T):
        basis = np.zeros((N, K))
        if not np.all(invalid_mask[i]):
            basis_ = compute_orth_basis(factors[i][~invalid_mask[i]])
            basis[~invalid_mask[i]] = basis_
        bases.append(basis)
    bases = np.stack(bases)

    data = np.nan_to_num(data)
    exposure = np.einsum("ijk,ij->ik", bases, data)
    projection = np.einsum("ijk,ik->ij", bases, exposure)
    return np.where(invalid_mask, np.nan, data - projection)


def _pandas_orth(
        data: pd.DataFrame,
        factors: Union[pd.DataFrame, Collection[pd.DataFrame]]) -> pd.DataFrame:
    """
    Orthogonalizes data by subtracting its projection onto the subspace spanned
    by the factor. The result is equivalent to the residuals of the regression of data
    on factors at each timestep.

    Parameters
    ----------
    data
        Time-series data.
    factors
        Factors. Should have a multi-index of time and factor.

    Returns
    -------
    Orthogonalized data.
    """
    if not isinstance(factors, pd.DataFrame):
        factors = stack(tuple(factors), names=["factor"])

    orth_data = _numpy_orth(data.values, pandas_to_numpy(factors))
    return pd.DataFrame(orth_data, index=data.index, columns=data.columns)


def orth(data, factors):
    """
    Orthogonalizes data by subtracting its projection onto the subspace spanned by
    the factors. The result is equivalent to regressing data on factors at each
    timestep and computing the residual

    Parameters
    ----------
    data
        Time-series data, e.g., a signal.
    factors
        Factors, e.g., one-hot country/ industry factor.

    Returns
    -------
    Orthogonalized data.
    """
    array_like = [data]
    if is_iterable_of(array_like, pd.DataFrame):
        return _pandas_orth(data, factors)
    elif is_iterable_of(array_like, np.ndarray):
        return _numpy_orth(data, factors)
    raise ValueError("data and factors must either both be dataframes or both be"
                     " numpy arrays")


def discretize(data: pd.DataFrame, q: Optional[int] = 5, axis=1,
               duplicates: str = "drop"):
    return data.apply(pd.qcut, axis=axis, q=q, labels=False, duplicates=duplicates)


def ts_zscore(data: pd.DataFrame, window_size: Optional[int] = None,
              min_periods: Optional[int] = None,
              fixed_mean: Optional[float] = None):
    """
    Computes rolling z-score of the data.

    Parameters
    ----------
    data
        Data.
    window_size
        Window size. If window size is None, an expanding window is used.
    min_periods
        Min periods.
    fixed_mean
        Fixed mean.

    Returns
    -------
    Z-scored data.

    Notes
    -----
    This can also be used for computing an expanding z-score by setting
    window_size to a value >= len(data).
    """
    # TODO: allow exponential moving average?
    if window_size is not None and min_periods is not None:
        # ensure min periods is less than window size
        min_periods = min(min_periods, window_size)

    if window_size is None:
        rolling = data.expanding(min_periods=min_periods)
    else:
        rolling = data.rolling(int(window_size), min_periods=min_periods)

    mean = fixed_mean
    if mean is None:
        mean = rolling.mean()

    return (data - mean) / rolling.std()


def xs_zscore(data: pd.DataFrame, fixed_mean: Optional[float] = None):
    mean = fixed_mean
    if mean is None:
        mean = data.mean(axis=1)
    return data.sub(mean, axis=0).div(data.std(axis=1), axis=0)


def xs_normalize(data: pd.DataFrame) -> pd.DataFrame:
    return data.div(data.sum(axis=1), axis=0)


def normalize(data: pd.Series) -> pd.Series:
    return data / data.sum()


def zscore(data: pd.DataFrame, window_size: Optional[int] = None,
           min_periods: Optional[int] = None,
           fixed_mean: Optional[float] = None, axis: int = 0):
    if axis not in [0, 1]:
        raise ValueError("Error computing z-score; axis must be 0 or 1")

    if axis == 0:
        return ts_zscore(data, window_size, min_periods, fixed_mean)
    return xs_zscore(data, fixed_mean)


def ffill(data: pd.DataFrame, limit: Optional[int] = None):
    if limit is None or limit > 0:
        data = data.ffill(limit=limit)
    return data


def shift(data: pd.DataFrame, periods: int):
    return data.shift(periods)


def clip(data: pd.DataFrame, threshold: Optional[float] = None):
    if threshold is not None:
        if threshold < 0:
            raise ValueError("Error clipping data; threshold must be >= 0")
        data = data.clip(lower=-threshold, upper=threshold)
    return data


# TODO: deprecate and use weighted_avg instead
def nan_aware_weighted_avg(objs, weights):
    shaper = objs[0]
    total_weights = pd.DataFrame(0, index=shaper.index, columns=shaper.columns)
    average = pd.DataFrame(0, index=shaper.index, columns=shaper.columns)
    for i, obj in enumerate(objs):
        average += obj.fillna(0) * weights[i]
        total_weights += (~obj.isnull()).astype(int) * weights[i]
    return average / total_weights


def expand_weights(weights: pd.DataFrame, times: Iterable[datetime]):
    """
    Gets weights for each column for each combination of source and time.

    Parameters
    ----------
    weights
        Dataframe with columns: start_time, end_time, key, column and weight.
    times
        Times to get weights for.

    Returns
    -------
    Expanded weights.
    """
    columns = weights["column"].unique()
    weights = map_ids(weights, pd.MultiIndex.from_product(
        [times, columns], names=["time", "column"]).to_frame(index=False), "column")
    weights = weights.set_index(["time", "key", "column"])["weight"]
    return weights.unstack().fillna(0)


# TODO: better name!
def compute_stack_avg(data: Union[Dict[str, pd.DataFrame], pd.DataFrame],
                      weights: Optional[Union[Dict[str, float], pd.Series, Dict[
                          str, pd.DataFrame], pd.DataFrame]] = None,
                      normalize_weights: bool = True, combine: bool = True):
    """
    Computes weighted average of stacked data.

    Parameters
    ----------
    data
        Data.
    weights
        Dataframe of weights. Must have multi-index of time and key, and columns
        matching the columns in each element of data. If weights are constant for
        each key and column over time, a dictionary/ series may alternatively be
        passed.
    normalize_weights
        Boolean indicating whether to normalize weights.
    combine
        Boolean indicating whether to sum the contributions from each source.

    Returns
    -------
    Average dataframe.
    """
    if isinstance(data, dict):
        data = stack(data)

    times = data.index.unique(level=0)
    keys = set(data.index.unique(level=1))
    columns = data.columns

    if isinstance(weights, pd.DataFrame):
        if weights.index.nlevels != 2:
            raise ValueError("Error computing stack average; if dataframe of weights"
                             " if passed, it must have a multi-index with time as the"
                             " first level and key as the second")

    if weights is None:
        weights = pd.Series(1, index=keys)
    if isinstance(weights, (dict, pd.Series)):
        weights = stack(
            {k: pd.DataFrame(weights[k], index=times, columns=columns) for k in
             weights.keys()})

    res = {}
    keys = keys.intersection(weights.index.unique(level=1))
    for k in keys:
        data_ = data.xs(k, level=1)
        # important to replace 0 with nan below because, in the case where we have
        # a non-zero weight for a nan entry and a zero weight for a non-nan entry,
        # we want the resulting entry to be nan rather than zero
        weights_ = weights.xs(k, level=1).replace(0, np.nan)
        # TODO: keep track of total non-nan weight like in nan_aware_weighted_avg?
        res[k] = data_ * weights_

    res = stack(res)
    # TODO: if normalize_weights, divide by total weights to ensure sum to 1
    if combine:
        # min_count=1 ensures entries nan for all keys are also nan in output
        res = res.groupby(level=0).sum(min_count=1)
    return res.reindex(columns=columns)


def deseason(data, groupby, window_size: int, min_periods: Optional[int] = None,
             include: Optional[Iterable] = None):
    """
    Deseasons data. Assumes data has mean of zero, e.g., returns.
    """
    adj = data.groupby(groupby).rolling(window_size, min_periods=min_periods).mean()
    adj = adj.swaplevel().sort_index()
    if include:
        adj = adj.loc[adj.index.get_level_values(1).isin(include)]
    # don't use data today to compute adjustment we will apply today
    adj = adj.groupby(level=1).shift().droplevel(1)
    return data - adj.reindex(index=data.index).fillna(0)


def merge_overlapping_intervals(intervals: pd.DataFrame):
    if intervals.empty:
        return intervals
    intervals = intervals.sort_values("start")
    return intervals.groupby(
        (intervals["start"] > intervals["end"].shift().cummax()).cumsum()).agg(
        {"start": "min", "end": "max"})


def interval_difference(intervals_1: pd.DataFrame,
                        intervals_2: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the intervals spanned by intervals_1 but not by intervals_2.

    Parameters
    ----------
    intervals_1
    intervals_2

    Returns
    -------
    Interval difference.
    """
    intervals_1 = merge_overlapping_intervals(intervals_1)
    intervals_2 = merge_overlapping_intervals(intervals_2)

    difference = []
    for i in range(len(intervals_1)):
        start = intervals_1.iloc[i]["start"]
        end = intervals_1.iloc[i]["end"]
        intervals_2_ = intervals_2[
            (intervals_2["end"] >= start) & (intervals_2["start"] <= end)]

        if intervals_2_.empty:
            difference.append((start, end))
            continue
        if start < intervals_2_.iloc[0]["start"]:
            difference.append((start, intervals_2_.iloc[0]["start"]))
        for j in range(len(intervals_2_) - 1):
            difference.append((intervals_2_.iloc[j]["end"],
                               intervals_2_.iloc[j + 1]["start"]))
        if end > intervals_2_.iloc[-1]["end"]:
            difference.append((intervals_2_.iloc[-1]["end"], end))
    return pd.DataFrame(difference, columns=["start", "end"])


def compute_drawdown(data: pd.DataFrame):
    return data / data.cummax() - 1.0


def compute_returns(data: Union[pd.DataFrame, pd.Series], log: bool = False) -> Union[
    pd.DataFrame, pd.Series]:
    if not log:
        return data.pct_change(fill_method=None)
    return np.log(data).diff()


# TODO: better name
def buffer_series(s: pd.Series, k: int) -> pd.Series:
    res = []
    value = None
    value_ = None
    count = 0
    for i in range(len(s)):
        if value is None:
            if not np.isnan(s.iloc[i]):
                value = s.iloc[i]
        elif ~np.isnan(s.iloc[i]) and s.iloc[i] != value:
            if value_ is None:
                value_ = s.iloc[i]
            if value_ == s.iloc[i]:
                count += 1
            else:
                value_ = s.iloc[i]
                count = 1
            if count == k:
                value = value_
        if value is None:
            res.append(np.nan)
        else:
            res.append(value)
    return pd.Series(res, index=s.index)


def minmax_norm(
        data: Union[float, pd.DataFrame, pd.Series], data_range: Tuple[float, float],
        target_range: Tuple[float, float] = (0, 1)) -> Union[
    float, pd.DataFrame, pd.Series]:
    return (data - data_range[0]) / (data_range[1] - data_range[0]) * (
            target_range[1] - target_range[0]) + target_range[0]
