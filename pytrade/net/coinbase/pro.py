from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pandas as pd
from pytrade.net.http import HttpRequest, HttpMethod, send_request

logger = logging.getLogger(__name__)

BASE_URL = "https://api.pro.coinbase.com"
TESTNET_BASE_URL = "https://api-public.sandbox.pro.coinbase.com"

CANDLES_COLUMNS = ["time", "low", "high", "open", "close", "volume"]

_CHUNK_SIZE_MAP = {
    timedelta(minutes=1): "3H",
    timedelta(minutes=5): "1D",
    timedelta(minutes=15): "3D",
    timedelta(hours=1): "10D",
    timedelta(hours=6): "2MS",
    timedelta(days=1): "6MS"
}


@dataclass
class Credentials:
    api_key: str
    api_secret: str
    passphrase: str


def _get_signature(api_secret: str, timestamp: str, method: HttpMethod,
                   endpoint: str) -> str:
    # TODO: make below work with POST data
    message = f"{timestamp}{str(method)}{endpoint}"
    h = hmac.new(base64.b64decode(api_secret), message.encode(), hashlib.sha256)
    return base64.b64encode(h.digest()).decode()


def auth_request(request: HttpRequest, credentials: Credentials):
    headers = {}
    if request.headers is not None:
        headers = request.headers.copy()
    timestamp = str(time.time())
    sign = _get_signature(credentials.api_secret, timestamp, request.method,
                          request.endpoint)
    headers["CB-ACCESS-KEY"] = credentials.api_key
    headers["CB-ACCESS-SIGN"] = sign
    headers["CB-ACCESS-TIMESTAMP"] = timestamp
    headers["CB-ACCESS-PASSPHRASE"] = credentials.passphrase
    request = replace(request, headers=headers)
    return request


def get_candles(product_id: str, granularity: str, start_time: datetime,
                end_time: datetime):
    """
    Gets candles for a symbol.

    Parameters
    ----------
    product_id
        Product to download candles for.
    granularity
        May be: 1min, 5min, 15min, 1h, 6h or 1D.
    start_time
        Start time (inclusive) to download data from. Note, this time is
        compared to the open time of each candle.
    end_time
        End time (exclusive) to download data to. Note, this time is compared
        to the open time of each candle.

    Returns
    -------
    Candles. The time for each candle represents the open time for the candle.

    Notes
    -----
    If downloading data for a candle which hasn't closed, that row of data will
    change until the candle has closed.
    """
    # TODO: reduce duplication with binance spot get_klines function
    data = []
    start_time = start_time.replace(tzinfo=timezone.utc)
    end_time = end_time.replace(tzinfo=timezone.utc)
    granularity = pd.Timedelta(granularity).to_pytimedelta()
    chunk_size = _CHUNK_SIZE_MAP[granularity]
    times = pd.date_range(start_time, end_time,
                          freq=chunk_size).to_pydatetime().tolist()
    if start_time not in times:
        times = [start_time] + times
    if end_time not in times:
        times += [end_time]
    for i in range(len(times) - 1):
        logger.info(f"Downloading data from: {times[i]} to {times[i + 1]}")
        chunk_start_time = times[i]
        chunk_end_time = times[i + 1] - timedelta(seconds=1)
        request = HttpRequest(method=HttpMethod.GET,
                              base_url=BASE_URL,
                              endpoint=f"/products/{product_id}/candles",
                              params={"start": str(int(chunk_start_time.timestamp())),
                                      "end": str(int(chunk_end_time.timestamp())),
                                      "granularity": str(
                                          int(granularity.total_seconds()))})
        res = send_request(request, tries=10, delay=20, max_delay=60 * 5,
                           backoff=2).json()
        if res:
            data.extend(res)
    data = pd.DataFrame(data, columns=CANDLES_COLUMNS)
    data["time"] = pd.to_datetime(data["time"], unit="s")
    data = data.set_index("time").astype(float)
    return data.sort_index()
