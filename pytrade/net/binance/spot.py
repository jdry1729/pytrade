from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import replace
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Union, Dict, Iterable, Optional
from urllib.parse import urlencode

import pandas as pd
from pandas.tseries.offsets import MonthEnd

from pytrade.data.utils import round_to_multiple
from pytrade.net.binance.common import Credentials, resolve_credentials
from pytrade.net.http import HttpRequest, HttpMethod, send_request
from pytrade.net.websocket import WebSocket
from pytrade.utils.pandas import stack

logger = logging.getLogger(__name__)

BASE_URL = "https://api.binance.com"
KLINES_COLUMNS = ['open_time', 'open', 'high', 'low', 'close', 'volume',
                  'close_time',
                  'quote_volume', 'count', 'taker_buy_volume',
                  'taker_buy_quote_volume', 'ignore']

FREQ_MAP = {
    "1w": "YS",
    "1M": "YS",
}

INTERVAL_TIMEDELTA_MAP = {
    "12h": timedelta(hours=12),
    "15m": timedelta(minutes=15),
    "1d": timedelta(days=1),
    "1h": timedelta(hours=1),
    "1m": timedelta(minutes=1),
    "1M": timedelta(weeks=4),
    "1w": timedelta(weeks=1),
    "2h": timedelta(hours=2),
    "30m": timedelta(minutes=30),
    "3d": timedelta(days=3),
    "3m": timedelta(minutes=3),
    "4h": timedelta(hours=4),
    "5m": timedelta(minutes=5),
    "6h": timedelta(hours=6),
    "8h": timedelta(hours=8),
}


def _get_chunk_size_from_interval(interval: timedelta):
    if interval >= timedelta(days=1):
        return "YS"
    elif interval > timedelta(hours=1):
        return "W-MON"
    elif interval > timedelta(minutes=4):
        return "D"
    return "6H"


class SideEffectType(Enum):
    NO_SIDE_EFFECT = 0
    MARGIN_BUY = 1
    AUTO_REPAY = 2


def _get_signature(api_secret: str, params: dict) -> str:
    query_string = urlencode(params)
    sign = hmac.new(api_secret.encode(), msg=query_string.encode(),
                    digestmod=hashlib.sha256).hexdigest()
    return sign


def auth_request(request: HttpRequest, credentials: Credentials):
    # copy params and headers so request isn't modified!
    params = {}
    headers = {}
    if request.params is not None:
        params = request.params.copy()
    if request.headers is not None:
        headers = request.headers.copy()
    timestamp = str(round(time.time() * 1000))
    params["timestamp"] = timestamp
    params["signature"] = _get_signature(credentials.api_secret, params)
    headers["X-MBX-APIKEY"] = credentials.api_key
    if request.method == HttpMethod.GET:
        headers["content-type"] = "application/json"
    elif request.method == HttpMethod.POST:
        headers["content-type"] = "application/x-www-form-urlencoded"
    request = replace(request, params=params, headers=headers)
    return request


async def subscribe(ws: WebSocket, channels: Union[str, Iterable[str]]):
    channels = [channels] if isinstance(channels, str) else list(set(channels))
    for channel in channels:
        logger.info(f"Subscribing to {channel}")
    message = {"method": "SUBSCRIBE", "params": channels, "id": ws.id}
    await ws.send(message)
    return message


async def unsubscribe(self, channels: Union[str, Iterable[str]]) -> Dict:
    channels = [channels] if isinstance(channels, str) else list(set(channels))
    for channel in channels:
        logger.info(f"Subscribing to {channel}")
    message = {"method": "UNSUBSCRIBE", "params": channels, "id": self.id}
    await self.send(message)
    return message


def get_klines(symbol: str, interval: str, start_time: datetime,
               end_time: datetime):
    """
    Parameters
    ----------
    symbol
    interval
        Must be one of: 12h, 15m, 1d, 1h, 1m, 1M, 1w, 2h, 30m,
        3d, 3m, 4h, 5m, 6h or 8h.
    start_time
        Start time (inclusive) to download data from.
    end_time
        End time (exclusive) to download data to.
    """
    data = []
    start_time = start_time.replace(tzinfo=timezone.utc)
    end_time = end_time.replace(tzinfo=timezone.utc)
    # use pd.TimeDelta to avoid this?
    interval_timedelta = INTERVAL_TIMEDELTA_MAP[interval]
    # we want chunk size to be as large as possible to reduce total
    # number of requests but must make sure number of klines returned for
    # each request is less than 400
    chunk_size = _get_chunk_size_from_interval(interval_timedelta)
    times = pd.date_range(start_time, end_time,
                          freq=chunk_size).to_pydatetime().tolist()
    if start_time not in times:
        times = [start_time] + times
    if end_time not in times:
        times += [end_time]
    for i in range(len(times) - 1):
        # subtract 1 from end time so exclusive
        # also set limit to 400 so request weight is 2 rather than 5
        params = {"symbol": symbol, "interval": interval,
                  "limit": 400,
                  "startTime": int(times[i].timestamp() * 1000),
                  "endTime": int(times[i + 1].timestamp() * 1000) - 1}
        request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                              endpoint="/api/v3/klines", params=params)
        res = send_request(request, tries=10, delay=20, max_delay=60 * 5,
                           backoff=2)
        data.extend(res.json())
    data = pd.DataFrame(data, columns=KLINES_COLUMNS)
    data = data.apply(pd.to_numeric, errors="ignore")
    # times are in utc time, maybe better to store timestamps
    data["open_time"] = pd.to_datetime(data["open_time"], unit="ms")
    data["close_time"] = pd.to_datetime(data["close_time"], unit="ms")
    return data.set_index("open_time")


def get_instruments():
    request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                          endpoint="/api/v3/exchangeInfo")
    res = send_request(request)
    data = pd.DataFrame(res.json()["symbols"])
    return data.set_index("symbol")


def get_instrument_created_time(symbol: str):
    month_klines = get_klines(symbol, interval="1M",
                              start_time=pd.Timestamp("2017-01-01"),
                              end_time=pd.Timestamp.now())
    month_start = month_klines.index[0]
    month_end = month_start + MonthEnd() + pd.Timedelta(days=1)
    day_klines = get_klines(symbol, interval="1d",
                            start_time=month_start,
                            end_time=month_end)
    return day_klines.index[0]


def get_instrument_filters():
    request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                          endpoint="/api/v3/exchangeInfo")
    res = send_request(request)
    filters = {x["symbol"]: pd.DataFrame(x["filters"]).set_index("filterType")
               for x in res.json()["symbols"]}
    filters = stack(filters.values(), filters.keys(), name="symbol")
    filters = filters.swaplevel().sort_index(level=0)
    return filters.apply(pd.to_numeric, errors="ignore")


def get_step_sizes():
    filters = get_instrument_filters()
    return filters.xs("LOT_SIZE", level=1)["stepSize"]


def get_accounts(credentials: Optional[Credentials] = None):
    # locked represents amount locked in limit orders
    # e.g., if you place a limit order to buy ETH using BUSD, limit price
    # * quanity of BUSD will be locked
    credentials = resolve_credentials(credentials)
    request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                          endpoint="/sapi/v1/margin/account",
                          params={"recvWindow": 10000})
    request = auth_request(request, credentials)
    res = send_request(request)
    accounts = pd.DataFrame(res.json()["userAssets"])
    accounts = accounts.apply(pd.to_numeric, errors="ignore")
    return accounts.set_index("asset")


def get_agg_account_info(credentials: Optional[Credentials] = None):
    credentials = resolve_credentials(credentials)
    request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                          endpoint="/sapi/v1/margin/account",
                          params={"recvWindow": 10000})
    request = auth_request(request, credentials)
    res = send_request(request).json()
    return pd.Series({
        "marginLevel": res["marginLevel"],
        "totalAssetOfBtc": res["totalAssetOfBtc"],
        "totalLiabilityOfBtc": res["totalLiabilityOfBtc"],
        "totalNetAssetOfBtc": res["totalNetAssetOfBtc"],
    })


def get_trades(symbol: str, start_time: datetime, end_time: datetime,
               credentials: Optional[Credentials] = None):
    credentials = resolve_credentials(credentials)
    request = HttpRequest(
        method=HttpMethod.GET,
        base_url=BASE_URL,
        endpoint="/sapi/v1/margin/myTrades",
        params={
            "symbol": symbol,
            "start_time": start_time.timestamp(),
            "end_time": end_time.timestamp(),
            "recvWindow": 10000,
        },
    )
    request = auth_request(request, credentials)
    res = send_request(request)
    trades = pd.DataFrame(res.json())
    return trades


def place_order(symbol: str, quantity: float, type_: str = "MARKET",
                *,
                limit_price: Optional[float] = None,
                time_in_force: Optional[str] = None,
                side_effect_type: SideEffectType =
                SideEffectType.NO_SIDE_EFFECT,
                credentials: Optional[Credentials] = None):
    credentials = resolve_credentials(credentials)
    step_size = get_step_sizes()[symbol]
    quantity = round(round_to_multiple(quantity, step_size), 5)
    order_str = f"{symbol} {quantity} {side_effect_type.name}"
    if quantity != 0:
        # TODO: check if quantity is 0
        logger.info(
            f"Placing order: {symbol} {quantity} {side_effect_type}")
        side = "BUY" if quantity > 0 else "SELL"
        params = {
            "symbol": symbol,
            "side": side,
            "type": type_,
            "quantity": abs(quantity),
            "sideEffectType": side_effect_type.name,
        }
        if limit_price is not None:
            params["price"] = limit_price
        if time_in_force is not None:
            params["timeInForce"] = time_in_force
        request = HttpRequest(
            method=HttpMethod.POST,
            endpoint="/sapi/v1/margin/order",
            params=params,
        )
        request = auth_request(request, credentials)
        try:
            res = send_request(request)
            res.raise_for_status()
        except Exception as e:
            logger.warning(
                f"Error placing order: {order_str}; {res.json()}")
            raise e
    else:
        logger.info(f"Not placing order: {order_str}; quantity is 0")
