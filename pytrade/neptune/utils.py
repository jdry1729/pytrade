import logging
from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd
from arcticdb_ext.exceptions import ArcticException
from pytrade.data import read_data

logger = logging.getLogger(__name__)


# TODO: could return None from read_data if no data found
def safe_read_data(library: str, symbol: str,
                   as_of: Optional[Union[int, str, datetime]] = None,
                   **kwargs) -> Optional:
    try:
        return read_data(library, symbol, as_of, **kwargs)
    except ArcticException:
        symbol = f"{symbol}:{library}"
        if as_of is not None:
            symbol += f"@{as_of}"
        logger.warning(f"Error reading {symbol}; not found")
        return None


def loc(obj, label, series_name: Optional[str] = None):
    if isinstance(obj, (tuple, list)):
        return tuple(loc(x, label) for x in obj)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        obj = obj.loc[label]
        if isinstance(obj, pd.Series):
            if series_name is not None:
                obj.name = series_name
    return obj


def slice_(obj, start: Optional = None, end: Optional = None):
    if isinstance(obj, (tuple, list)):
        return tuple(slice_(x, start, end) for x in obj)
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        if start is not None:
            obj = obj.loc[start:]
        if end is not None:
            obj = obj.loc[:end]
    return obj


def get_as_live(live_values: pd.Series, live_time: datetime,
                log: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """
    Gets as-live values.

    Parameters
    ----------
    live_values
        Live values. Must have index with name "stock_id".
    live_time
        Live time.
    log
        Logged live values.

    Returns
    -------
    As-live values.
    """
    as_live = pd.DataFrame(
        [live_values], index=pd.Index([live_time], name="time"))
    if log is not None:
        log = log.loc[:live_time - timedelta(seconds=1)]
        log = log.groupby(["_time", "stock_id"]).last().unstack()
        log.index.name = "time"
        return pd.concat([log, as_live])
    return as_live
