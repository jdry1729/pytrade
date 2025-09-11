from typing import Iterable

import pandas as pd
from pytrade.utils.functions import partial
from pytrade.utils.pandas import ensure_index_names


def _maybe_group(s: pd.Series, by: Iterable[str]):
    if by:
        return s.reset_index(level=by).groupby(by)["value"]
    return s


def compute_rolling_mean(
        events: pd.Series, resample_rule: str = "1D", window: int = 90
):
    """
    Computes rolling mean given a series of "events". If events is multi-indexed,
    rolling mean will be computed group-wise.

    Parameters
    ----------
    events
        Events to compute rolling mean for. Index must be datetime index if singly
        indexed. If multi-indexed, first level must be time, and all other levels are
        grouped on.
    resample_rule
        Period frequency. Determines frequency of output series.
    window
        Number of periods to use to compute mean.

    Returns
    -------
    Rolling mean.
    """
    events = events.rename("value")
    nlevels = events.index.nlevels
    events.index = ensure_index_names(events.index, levels=range(1, nlevels))

    group_fn = partial(_maybe_group, by=events.index.names[1:])
    res = group_fn(events).resample(resample_rule)
    sum_, count = [group_fn(x).rolling(window).sum() for x in [res.sum(), res.count()]]
    mean = (sum_ / count).rename("mean")

    if nlevels > 1:
        mean = mean.reorder_levels([nlevels - 1] + list(range(nlevels - 1)))
    return mean.sort_index()


# TODO: reuse code with above function
def compute_rolling_count(events: pd.Series, resample_rule: str = "1D",
                          window: int = 90):
    """"
    Computes rolling count given a series of "events". If events is multi-indexed,
    rolling count will be computed group-wise.
    """
    events = events.rename("value")
    nlevels = events.index.nlevels
    events.index = ensure_index_names(events.index, levels=range(1, nlevels))

    group_fn = partial(_maybe_group, by=events.index.names[1:])
    res = group_fn(events).resample(resample_rule)
    count = group_fn(res.count()).rolling(window).sum().rename("count")

    if nlevels > 1:
        count = count.reorder_levels([nlevels - 1] + list(range(nlevels - 1)))
    return count.sort_index()
