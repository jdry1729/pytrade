import logging
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional

import pandas as pd
from tqdm import tqdm

from pytrade.net.http import HttpRequest, _send_request
from pytrade.utils.time import get_equally_spaced_times

logger = logging.getLogger(__name__)

VISION_URL = "https://data.binance.vision"


def get_historical_csv_data(exchange: str, symbol: str, data_type_: str,
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            interval: Optional[str] = None):
    """
    Gets historical CSV data from Binance Vision.
    """
    data = []
    path = f"/data/{exchange}/daily/{data_type_}/{symbol}"
    if interval is not None:
        path += f"/{interval}"
    times = get_equally_spaced_times(start_time, end_time, period=timedelta(days=1))
    for time in tqdm(times):
        time_str = time.strftime("%Y-%m-%d")
        logger.debug(f"Downloading {symbol} {data_type_} data for: {time_str}")
        req = HttpRequest(base_url=VISION_URL,
                          endpoint=f"{path}/{symbol}-{data_type_}-{time_str}.zip")
        try:
            res = _send_request(req)
        except Exception:
            logger.warning(f"Error getting {symbol} {data_type_} data for: {time_str};"
                           f" file not found")
        else:
            data.append(pd.read_csv(BytesIO(res.content), compression="zip"))
    return pd.concat(data, axis=0)
