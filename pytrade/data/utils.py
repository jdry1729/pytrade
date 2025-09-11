import numpy as np
import pandas as pd


def round_to_multiple(n, step):
    if n >= 0:
        return step * (n // step)
    return - step * (- n // step)


def get_const_intervals(s: pd.Series) -> pd.DataFrame:
    """
    Gets constant intervals of a series.

    Parameters
    ----------
    s
        Series. Index must have an ordering.

    Returns
    -------
    DataFrame with constant intervals. Start and end columns are inclusive.
    """
    res = []
    changes = s[(s != s.shift()) | (s != s.shift(-1))]
    start = None
    value = None
    N = len(changes)
    for i in range(N):
        if start is None:
            start = changes.index[i]
            value = changes.iloc[i]
            if i == N - 1:
                res.append([start, start, value])
        else:
            if changes.iloc[i] == value:
                res.append([start, changes.index[i], value])
                start = None
            else:
                res.append([start, start, value])
                start = changes.index[i]
                value = changes.iloc[i]
                if i == N - 1:
                    res.append([start, start, value])
    return pd.DataFrame(res, columns=["start", "end", "value"])


def alpha_to_halflife(alpha: float) -> float:
    """
    Notes
    -----
    This can be used to get hallife associated with particular alpha
    used in pandas EWMA function.
    """
    return - np.log(2) / np.log(1 - alpha)


def halflife_to_alpha(halflife: float) -> float:
    return 1 - 0.5 ** (1.0 / halflife)


def halflife_to_lambda(half_life: float) -> float:
    return np.log(2) / half_life


def leading_zeros_to_nan(data: pd.Series) -> pd.Series:
    mask = data.abs().gt(0)
    if mask.any():
        first_valid_loc = mask[mask].index[0]
        first_valid_iloc = data.index.get_loc(first_valid_loc)
        data.iloc[:first_valid_iloc] = np.nan
    else:
        data.loc[:] = np.nan
    return data


def remove_leading_nans(data: pd.Series) -> pd.Series:
    first_valid_index = data.first_valid_index()
    if first_valid_index is not None:
        return data.loc[first_valid_index:]
    raise data.iloc[len(data):]


def get_value_streak(x: pd.Series) -> pd.Series:
    x = x.ne(x.shift()).cumsum()
    return x.groupby(x).cumcount() + 1


def rankings_to_volume(rankings: pd.DataFrame, halflife: float = 100e3,
                       max_volume: float = 100) -> pd.DataFrame:
    return max_volume * np.exp(-np.log(2) * rankings / halflife)
