from typing import Optional, Literal

import numpy as np
import pandas as pd


def event_decay_signal(events: pd.Series, lambda_: float, times: pd.Index,
                       min_value: Optional[float] = None,
                       max_periods: Optional[int] = None,
                       mode: Literal["sum", "max"] = "sum",
                       fill_value: float = 0.0) -> pd.Series:
    T = len(times)
    locs = np.arange(T)

    out = np.full(T, fill_value, dtype=float)
    for time, v in events.replace(0, np.nan).dropna().items():
        # TODO: get next loc if time not in times
        start = times.get_loc(time)
        end = None if max_periods is None else start + max_periods
        values = v * np.exp(-lambda_ * (locs[start:end] - start))
        if mode == "sum":
            out[start:end] += values
        elif mode == "max":
            out[start:end] = np.maximum(out[start:end], values)
    out = pd.Series(out, index=times)
    if min_value is not None:
        out = out.where(out.abs() >= min_value, 0)
    return out
