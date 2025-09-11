import logging
from datetime import datetime, timedelta
from typing import Union, Literal, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from pytrade.net.http import HttpRequest, send_request_cffi
from pytrade.utils.retry import retry
from pytrade.utils.time import get_equally_spaced_times
from tqdm import tqdm

BASE_URL = "https://tvc6.investing.com/{uuid}/0/0/0/0"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like"
        " Gecko) Chrome/104.0.5112.102 Safari/537.36"
    ),
    "Referer": "https://tvc-invdn-com.investing.com/",
    "Content-Type": "application/json",
}

logger = logging.getLogger(__name__)

# TODO: increase chunk sizes below?
CHUNK_SIZE = {
    1: timedelta(days=1),
    5: timedelta(days=5),
    15: timedelta(days=10),
    30: timedelta(days=30),
    60: timedelta(days=100),
    300: timedelta(days=200),
    "D": timedelta(days=300),
    "W": timedelta(days=300),
    "M": timedelta(days=600),
}


def _send_request(request: HttpRequest):
    res = send_request_cffi(request, raise_for_status=False)
    try:
        res.raise_for_status()
        return res
    except Exception:
        raise Exception(f"{res.status_code} error sending request; {res.content}")


def search(query: str, limit: int = 10, type_: str = None,
           exchange: Union[str, None] = None):
    if type_ is None:
        type_ = ""
    if exchange is None:
        exchange = ""
    params = {
        "query": query,
        "limit": limit,
        "type": type_,
        "exchange": exchange,
    }
    request = HttpRequest(base_url=BASE_URL.format(uuid=uuid4().hex),
                          endpoint="/search", params=params, headers=HEADERS)
    res = retry(_send_request, args=(request,)).json()
    return pd.DataFrame(res)


def get_olhcv_data(symbol: int, start_time: datetime, end_time: datetime,
                   resolution: Union[str, int]):
    """
    Gets OHLCV data. Times are in UTC.
    """
    data = []
    times = get_equally_spaced_times(start_time, end_time,
                                     period=CHUNK_SIZE[resolution])
    if end_time not in times:
        times.append(end_time)

    for i in tqdm(range(len(times) - 1)):
        chunk_start_time = times[i]
        chunk_end_time = times[i + 1]

        params = {
            "symbol": symbol,
            "from": int(chunk_start_time.timestamp()),
            "to": int(chunk_end_time.timestamp()),
            "resolution": resolution,
        }
        request = HttpRequest(base_url=BASE_URL.format(uuid=uuid4().hex),
                              endpoint="/history", params=params, headers=HEADERS)
        res = retry(_send_request, args=(request,)).json()
        status = res.pop("s")
        if status == "ok":
            data.append(pd.DataFrame(res))
        elif status == "no_data":
            logger.debug(f"No data found for {symbol} between {start_time} and"
                         f" {end_time}")
    data = pd.concat(data, axis=0)
    data["t"] = pd.to_datetime(data["t"], unit="s")
    data = data.replace("n/a", np.nan)
    data = data.set_index("t")
    data = data[~data.index.duplicated(keep="last")]
    return data
