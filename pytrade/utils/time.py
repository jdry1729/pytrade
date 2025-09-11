import logging
import random
import time as time_
from datetime import datetime, timedelta, date, time
from typing import Union, Tuple, List

import pandas as pd

logger = logging.getLogger(__name__)

ISO_8601_FORMAT = "%Y-%m-%dT%H:%M:%S"


def normalize(d: datetime):
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def sleep(secs: Union[int, Tuple[int, int]]) -> None:
    if isinstance(secs, int) and secs == 0:
        return
    if isinstance(secs, tuple):
        secs = random.uniform(*secs)
    logger.debug(f"Sleeping for {secs:.2f}s")
    time_.sleep(secs)


def date_to_datetime(d: date) -> datetime:
    return datetime.combine(d, time())


def sleep_until(time: datetime):
    # time must be UTC!
    now = datetime.utcnow()
    secs = (time - now).total_seconds()
    if secs > 0:
        time_.sleep(secs)


def time_range(start_time: datetime, end_time: datetime,
               freq: str, normalize: bool = False) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.date_range(start_time, end_time, freq=freq,
                                          normalize=normalize),
                            name="time")


def get_equally_spaced_times(start_time: datetime, end_time: datetime,
                             period: timedelta) -> List[datetime]:
    return pd.date_range(start_time, end_time, freq=period).to_pydatetime().tolist()
