import logging
from datetime import datetime
from datetime import timezone

import pandas as pd
from pytrade.net.binance.utils import get_historical_csv_data
from pytrade.net.http import HttpRequest, _send_request
from pytrade.utils.retry import retry
from pytrade.utils.time import get_equally_spaced_times
from tqdm import tqdm

BASE_URL = "https://fapi.binance.com"

# max rows returned by binance is 500
CHUNK_SIZE = {
    "1d": pd.Timedelta("500D"),
    "5m": pd.Timedelta("1D")
}

logger = logging.getLogger(__name__)

_LONG_SHORT_RATIO_ENDPOINT_MAP = {
    "top_account": "/futures/data/topLongShortAccountRatio",
    "top_position": "/futures/data/topLongShortPositionRatio",
    "global_account": "/futures/data/globalLongShortAccountRatio",
}


def get_exchange_info():
    request = HttpRequest(base_url=BASE_URL,
                          endpoint="/fapi/v1/exchangeInfo")
    res = retry(_send_request, args=(request,), max_tries=3)
    return res.json()


def get_long_short_ratio(symbol: str, period: str, start_time: datetime,
                         end_time: datetime, type_: str = "top_account"):
    data = []
    start_time = start_time.replace(tzinfo=timezone.utc)
    end_time = end_time.replace(tzinfo=timezone.utc)
    times = get_equally_spaced_times(start_time, end_time, period=CHUNK_SIZE[period])
    if end_time not in times:
        times.append(end_time)
    endpoint = _LONG_SHORT_RATIO_ENDPOINT_MAP[type_]
    for i in tqdm(range(len(times) - 1)):
        logger.debug(f"Downloading data from: {times[i]} to {times[i + 1]}")
        params = {"symbol": symbol, "period": period,
                  "limit": 500,
                  "startTime": int(times[i].timestamp() * 1000),
                  "endTime": int(times[i + 1].timestamp() * 1000) - 1}
        request = HttpRequest(base_url=BASE_URL,
                              endpoint=endpoint,
                              params=params)
        res = retry(_send_request, args=(request,), max_tries=3)
        data.extend(res.json())
    data = pd.DataFrame(data)
    data["time"] = pd.to_datetime(data["timestamp"], unit="ms")
    data = data.set_index("time")
    data = data.drop(columns=["timestamp", "symbol"])
    data = data.apply(pd.to_numeric, errors="ignore").astype(float)
    return pd.DataFrame(data)


def get_historical_metrics(symbol: str, start_time: datetime, end_time: datetime):
    data = get_historical_csv_data("futures/um", symbol, "metrics", start_time,
                                   end_time)
    data["create_time"] = pd.to_datetime(data["create_time"])
    data = data.set_index("create_time")
    return data.drop(columns="symbol")
