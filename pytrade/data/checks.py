import logging
from datetime import datetime, timedelta
from typing import Union, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def raise_if_stale(data: Union[pd.DataFrame, pd.Series],
                   max_staleness: timedelta,
                   time: Optional[datetime] = None):
    data = data.dropna(how="all")
    latest_time = data.index.get_level_values(0).max()
    time_diff = time - latest_time
    if time - latest_time > max_staleness:
        raise ValueError(f"Data stale by {time_diff}; this exceeds max"
                         f" staleness of {max_staleness}")
