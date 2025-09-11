from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone, timedelta
from typing import Optional, Union, Dict, Iterable, List

import pandas as pd
from pytrade.net.http import HttpRequest, HttpMethod, send_request
from pytrade.net.websocket import WebSocket
from pytrade.utils.pandas import stack

logger = logging.getLogger(__name__)

# TOOD: just have one base url!
BASE_URL = "https://futures.kraken.com/derivatives"
BASE_URL_2 = "https://futures.kraken.com"
CHARTS_BASE_URL = "https://futures.kraken.com/api/charts/v1/"

KLINES_COLUMNS = ["open", "high", "low", "close", "volume", "time"]

ACCOUNT_LOG_CHUNK_SIZE = 10000

POSITIONS_COLUMNS = [
    "fillTime",
    "maxFixedLeverage",
    "pnlCurrency",
    "price",
    "side",
    "size",
    "symbol",
    "unrealizedFunding"
]

RESOLUTION_TIMEDELTA_MAP = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
}


def _get_chunk_size_from_resolution(resolution: timedelta):
    if resolution >= timedelta(days=1):
        return "YS"
    elif resolution >= timedelta(hours=1):
        return "6MS"
    elif resolution >= timedelta(minutes=5):
        return "W-MON"
    return "1D"


@dataclass
class Credentials:
    api_key: str
    api_secret: str


@dataclass
class WsAuthConfig:
    api_key: str
    original_challenge: str
    signed_challenge: str


def _get_signature(secret_key: str, message: str) -> str:
    h = hmac.new(
        base64.b64decode(secret_key),
        hashlib.sha256(message.encode()).digest(),
        hashlib.sha512,
    )
    return base64.b64encode(h.digest()).decode()


async def subscribe(ws: WebSocket, channels: Union[str, Iterable[str]],
                    auth_config: Optional[WsAuthConfig] = None) -> List[dict]:
    """
    Subscribe to channels. You must have authenticated the connection prior
    to subscribing.
    """
    channels = [channels] if isinstance(channels, str) else list(set(channels))
    messages = _get_sub_messages(channels, auth_config)
    for message in messages:
        for channel in message.pop("channels"):
            logger.info(f"Subscribing to {channel}")
        await ws.send(message)
    return messages


async def unsubscribe(ws: WebSocket, channels: Union[str, Iterable[str]],
                      auth_config: Optional[WsAuthConfig] = None) -> \
        List[dict]:
    """
    Unsubscribe from channels.
    """
    channels = [channels] if isinstance(channels, str) else list(set(channels))
    messages = _get_sub_messages(channels, auth_config, unsub=True)
    for message in messages:
        for channel in message.pop("channels"):
            logger.info(f"Unsubscribing from {channel}")
        await ws.send(message)
    return messages


async def auth_ws(ws: WebSocket, credentials: Credentials) -> WsAuthConfig:
    if credentials.api_key is not None:
        await ws.send({"event": "challenge", "api_key": credentials.api_key})
        # TODO: implement timeout/ retries here
        async for message in ws:
            if message.get("event") == "challenge":
                # binding challenges to ws below lets us avoid having to pass
                # them to subscribe functions
                original_challenge = message["message"]
                signed_challenge = _get_signature(credentials.api_secret,
                                                  original_challenge)
                logger.info("Authenticated to Kraken Futures WebSocket API")
                return WsAuthConfig(api_key=credentials.api_key,
                                    original_challenge=original_challenge,
                                    signed_challenge=signed_challenge)


def _get_sub_messages(channels: List[str],
                      auth_config: Optional[WsAuthConfig] = None,
                      unsub: bool = False) -> List[
    Dict]:
    """
    Get subscription messages for channels.
    """
    feeds = set(channel.split(":")[0] for channel in channels)
    prefix = "un" if unsub else ""
    messages = []
    for feed in feeds:
        _channels = [
            channel for channel in channels if channel.split(":")[0] == feed
        ]
        if feed in ["ticker", "trade", "book", "ticker_lite"]:
            product_ids = [_channel.split(":")[1] for _channel in _channels]
            message = {
                "event": f"{prefix}subscribe",
                "feed": feed,
                "product_ids": product_ids,
            }
        else:
            message = {
                "event": f"{prefix}subscribe",
                "feed": feed,
            }
        if auth_config is not None:
            message["api_key"] = auth_config.api_key
            message["original_challenge"] = auth_config.original_challenge
            message["signed_challenge"] = auth_config.signed_challenge
        message["channels"] = _channels
        messages.append(message)
    return messages


def get_post_data(data: Dict) -> str:
    return "&".join(f"{k}={v}" for k, v in data.items())


def auth_request(request: HttpRequest, credentials: Credentials) -> HttpRequest:
    # copy headers so request isn't modified!
    headers = {}
    if request.headers is not None:
        headers = request.headers.copy()
    # assume POST data is passed to HttpRequest through json arg
    post_data = ""
    if request.params is not None:
        post_data = get_post_data(request.params)
    nonce = int(1000 * time.time())
    message = f"{post_data}{nonce}{request.endpoint}"
    sign = _get_signature(credentials.api_secret, message)
    headers["APIKey"] = credentials.api_key
    headers["Nonce"] = str(nonce)
    headers["Authent"] = sign
    request = replace(request, headers=headers, data=post_data)
    return request


def get_instruments():
    request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                          endpoint="/api/v3/instruments")
    res = send_request(request)
    data = pd.DataFrame(res.json()["instruments"])
    data = data[
        ["category", "contractSize", "lastTradingTime", "maxPositionSize",
         "openingDate", "symbol", "type", "underlying",
         "contractValueTradePrecision"]]
    return data.set_index("symbol")


def get_candles(symbol: str, resolution: str, tick_type: str,
                start_time: datetime, end_time: datetime):
    """
    Gets candles for a symbol.

    Parameters
    ----------
    symbol
        Symbol to download candles for.
    resolution
        May be: "1m", "5m", "15m", "30m", "1h", "4h", "12h", "1d", "1w".
    tick_type
        May be "spot", "mark" or "trade".
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
    There's a bug with Kraken's API which causes candle with open time equal to "to"
    timestamp not to be downloaded if no trades occurred in the period associated
    with the final candle. To guard against this, to get data up until 20:30, for
    example, on a particular day, set the end_time argument to 20:30:02.
    """
    # TODO: reduce duplication with binance spot get_klines function
    data = []
    start_time = start_time.replace(tzinfo=timezone.utc)
    end_time = end_time.replace(tzinfo=timezone.utc)
    resolution_timedelta = RESOLUTION_TIMEDELTA_MAP[resolution]
    chunk_size = _get_chunk_size_from_resolution(resolution_timedelta)
    times = pd.date_range(start_time, end_time,
                          freq=chunk_size).to_pydatetime().tolist()
    if start_time not in times:
        times = [start_time] + times
    if end_time not in times:
        times += [end_time]
    for i in range(len(times) - 1):
        logger.info(f"Downloading data from: {times[i]} to {times[i + 1]}")
        chunk_start_time = times[i]
        # subtract second from end time so it's exclusive
        chunk_end_time = times[i + 1] - timedelta(seconds=1)
        # request below gets candles where the open time is greater than or
        # equal to chunk start time and less than or equal to chunk end time
        request = HttpRequest(method=HttpMethod.GET,
                              base_url=CHARTS_BASE_URL,
                              endpoint=f"{tick_type}/{symbol}/{resolution}",
                              params={"from": int(chunk_start_time.timestamp()),
                                      "to": int(chunk_end_time.timestamp())})
        res = send_request(request, tries=-1, delay=20, max_delay=60 * 5,
                           backoff=2).json()
        candles = res["candles"]
        if candles:
            data.extend(res["candles"])
    data = pd.DataFrame(data, columns=KLINES_COLUMNS)
    data["time"] = pd.to_datetime(data["time"], unit="ms")
    data = data.set_index("time")
    data = data.apply(pd.to_numeric, errors="ignore").astype(float)
    return data


def get_flex_margin_account(credentials: Credentials):
    request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                          endpoint="/api/v3/accounts")
    request = auth_request(request, credentials)
    data = send_request(request).json()
    data = data["accounts"]["flex"]
    data = {k: v for k, v in data.items() if k in [
        "initialMargin",
        "initialMarginWithOrders",
        "maintenanceMargin",
        "balanceValue",
        "portfolioValue",
        "collateralValue",
        "pnl",
        "unrealizedFunding",
        "totalUnrealized",
        "totalUnrealizedAsMargin",
        "availableMargin",
        "marginEquity"
    ]}
    return pd.Series(data)


def get_open_positions(credentials: Credentials):
    request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                          endpoint="/api/v3/openpositions")
    request = auth_request(request, credentials)
    res = send_request(request).json()
    return pd.DataFrame(res["openPositions"], columns=POSITIONS_COLUMNS)


def get_historical_funding_rates(symbols: Iterable[str]):
    """
    Gets historical funding rates.

    Notes
    -----
    The row corresponding to hour T represents the funding rate to be paid between
    hours T and T + 1.
    """
    rates = {}
    for symbol in symbols:
        request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                              endpoint="/api/v4/historicalfundingrates",
                              params={"symbol": symbol})
        res = send_request(request).json()
        rates_ = pd.DataFrame(res["rates"])
        rates_["timestamp"] = pd.to_datetime(rates_["timestamp"]).dt.tz_localize(None)
        rates[symbol] = rates_.set_index("timestamp")
    return stack(rates, names="symbol")


def _get_account_log(credentials: Credentials, *, since: Optional[datetime] = None,
                     before: Optional[datetime] = None, from_: Optional[int] = None,
                     to_: Optional[int] = None, count: int = 500, sort: str = "asc"):
    params = {"count": count, "sort": sort}
    if since is not None:
        params["since"] = int(since.timestamp() * 1000)
    if before is not None:
        params["before"] = int(before.timestamp() * 1000)
    if from_ is not None:
        params["from"] = from_
    if to_ is not None:
        params["to"] = to_
    request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL_2,
                          endpoint="/api/history/v3/account-log", params=params)
    request = auth_request(request, credentials)
    res = send_request(request).json()
    return pd.DataFrame(res["logs"])


def get_account_log(credentials: Credentials, start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None):
    start_id = _get_account_log(credentials, since=start_time, count=1,
                                sort="asc").iloc[0]["id"]
    end_id = _get_account_log(credentials, before=end_time, count=1,
                              sort="desc").iloc[0]["id"]

    logs = []
    for i in range(start_id, end_id, ACCOUNT_LOG_CHUNK_SIZE):
        logs_ = _get_account_log(credentials, from_=i,
                                 to_=min(i + ACCOUNT_LOG_CHUNK_SIZE, end_id),
                                 count=ACCOUNT_LOG_CHUNK_SIZE)
        logs.append(logs_)
    logs = pd.concat(logs).set_index("id")
    logs["date"] = pd.to_datetime(logs["date"], format="mixed").dt.tz_localize(None)
    return logs


def get_account_curve_from_log(account_log: pd.DataFrame):
    """
    Gets account curve, doesn't account for unrealized PNL.
    """
    account_log = account_log.set_index("date")
    pnl = account_log["realized_pnl"].fillna(0).resample(
        "1D", closed="right", label="right").sum()
    funding = account_log["realized_funding"].fillna(0).resample(
        "1D", closed="right", label="right").sum()
    fees = account_log["fee"].fillna(0).resample(
        "1D", closed="right", label="right").sum()
    return pnl + funding - fees


def _get_execution_events(credentials: Credentials, *, since: Optional[datetime] = None,
                          before: Optional[datetime] = None, count: int = 500,
                          sort: str = "asc"):
    params = {"count": count, "sort": sort}
    if since is not None:
        params["since"] = int(since.timestamp() * 1000)
    if before is not None:
        params["before"] = int(before.timestamp() * 1000)
    request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL_2,
                          endpoint="/api/history/v3/executions", params=params)
    request = auth_request(request, credentials)
    res = send_request(request).json()
    return res


def get_slippage(symbol, resolution: str, start_time: datetime, end_time: datetime):
    data = []
    interval = pd.Timedelta(resolution).total_seconds()
    start_time = start_time.replace(tzinfo=timezone.utc)
    end_time = end_time.replace(tzinfo=timezone.utc)
    resolution_timedelta = RESOLUTION_TIMEDELTA_MAP[resolution]
    chunk_size = _get_chunk_size_from_resolution(resolution_timedelta)
    times = pd.date_range(start_time, end_time,
                          freq=chunk_size).to_pydatetime().tolist()
    if start_time not in times:
        times = [start_time] + times
    if end_time not in times:
        times += [end_time]
    for i in range(len(times) - 1):
        logger.info(f"Downloading data from: {times[i]} to {times[i + 1]}")
        chunk_start_time = times[i]
        # subtract second from end time so it's exclusive
        chunk_end_time = times[i + 1] - timedelta(seconds=1)
        request = HttpRequest(method=HttpMethod.GET,
                              base_url=CHARTS_BASE_URL,
                              endpoint=f"analytics/{symbol}/slippage",
                              params={"since": int(chunk_start_time.timestamp()),
                                      "to": int(chunk_end_time.timestamp()),
                                      "interval": int(interval)})
        res = send_request(request, tries=-1, delay=20, max_delay=60 * 5,
                           backoff=2).json()["result"]
        index = pd.to_datetime(res["timestamp"], unit="s")
        bid_slippage = pd.DataFrame(res["data"]["bid"], index=index)
        ask_slippage = pd.DataFrame(res["data"]["ask"], index=index)
        data.append(stack([ask_slippage, bid_slippage], ["ask", "bid"], axis=1))

    data = pd.concat(data).sort_index()
    return data.apply(pd.to_numeric, errors="ignore").astype(float)


def get_best_bid_and_ask(symbol, resolution: str, start_time: datetime,
                         end_time: datetime):
    data = []
    interval = pd.Timedelta(resolution).total_seconds()
    start_time = start_time.replace(tzinfo=timezone.utc)
    end_time = end_time.replace(tzinfo=timezone.utc)
    resolution_timedelta = RESOLUTION_TIMEDELTA_MAP[resolution]
    chunk_size = _get_chunk_size_from_resolution(resolution_timedelta)
    times = pd.date_range(start_time, end_time,
                          freq=chunk_size).to_pydatetime().tolist()
    if start_time not in times:
        times = [start_time] + times
    if end_time not in times:
        times += [end_time]
    for i in range(len(times) - 1):
        logger.info(f"Downloading data from: {times[i]} to {times[i + 1]}")
        chunk_start_time = times[i]
        # subtract second from end time so it's exclusive
        chunk_end_time = times[i + 1] - timedelta(seconds=1)
        request = HttpRequest(method=HttpMethod.GET,
                              base_url=CHARTS_BASE_URL,
                              endpoint=f"analytics/{symbol}/spreads",
                              params={"since": int(chunk_start_time.timestamp()),
                                      "to": int(chunk_end_time.timestamp()),
                                      "interval": int(interval)})
        res = send_request(request, tries=-1, delay=20, max_delay=60 * 5,
                           backoff=2).json()["result"]
        index = pd.to_datetime(res["timestamp"], unit="s")
        best_bid = pd.Series(res["data"]["bid"]["best_price"], index=index)
        best_ask = pd.Series(res["data"]["ask"]["best_price"], index=index)
        data.append(pd.concat([best_bid.rename("best_bid"),
                               best_ask.rename("best_ask")], axis=1))

    data = pd.concat(data).sort_index()
    return data.apply(pd.to_numeric, errors="ignore").astype(float)


# TODO: have one method for all get analytic functions
def get_long_short_ratio(symbol, resolution: str, start_time: datetime,
                         end_time: datetime):
    data = []
    interval = pd.Timedelta(resolution).total_seconds()
    start_time = start_time.replace(tzinfo=timezone.utc)
    end_time = end_time.replace(tzinfo=timezone.utc)
    resolution_timedelta = RESOLUTION_TIMEDELTA_MAP[resolution]
    chunk_size = _get_chunk_size_from_resolution(resolution_timedelta)
    times = pd.date_range(start_time, end_time,
                          freq=chunk_size).to_pydatetime().tolist()
    if start_time not in times:
        times = [start_time] + times
    if end_time not in times:
        times += [end_time]
    for i in range(len(times) - 1):
        logger.info(f"Downloading data from: {times[i]} to {times[i + 1]}")
        chunk_start_time = times[i]
        # subtract second from end time so it's exclusive
        chunk_end_time = times[i + 1] - timedelta(seconds=1)
        request = HttpRequest(method=HttpMethod.GET,
                              base_url=CHARTS_BASE_URL,
                              endpoint=f"analytics/{symbol}/long-short-ratio",
                              params={"since": int(chunk_start_time.timestamp()),
                                      "to": int(chunk_end_time.timestamp()),
                                      "interval": int(interval)})
        res = send_request(request, tries=-1, delay=20, max_delay=60 * 5,
                           backoff=2).json()["result"]
        long_short_ratio = pd.Series(res["data"], index=pd.to_datetime(res["timestamp"],
                                                                       unit="s"))
        data.append(long_short_ratio)

    data = pd.concat(data).sort_index()
    return data.apply(pd.to_numeric, errors="ignore").astype(float)


# TODO: have one method for all kraken analytics functions
def get_aggressor_differential(symbol, resolution: str, start_time: datetime,
                               end_time: datetime):
    data = []
    interval = pd.Timedelta(resolution).total_seconds()
    start_time = start_time.replace(tzinfo=timezone.utc)
    end_time = end_time.replace(tzinfo=timezone.utc)
    resolution_timedelta = RESOLUTION_TIMEDELTA_MAP[resolution]
    chunk_size = _get_chunk_size_from_resolution(resolution_timedelta)
    times = pd.date_range(start_time, end_time,
                          freq=chunk_size).to_pydatetime().tolist()
    if start_time not in times:
        times = [start_time] + times
    if end_time not in times:
        times += [end_time]
    for i in range(len(times) - 1):
        logger.info(f"Downloading data from: {times[i]} to {times[i + 1]}")
        chunk_start_time = times[i]
        # subtract second from end time so it's exclusive
        chunk_end_time = times[i + 1] - timedelta(seconds=1)
        request = HttpRequest(method=HttpMethod.GET,
                              base_url=CHARTS_BASE_URL,
                              endpoint=f"analytics/{symbol}/aggressor-differential",
                              params={"since": int(chunk_start_time.timestamp()),
                                      "to": int(chunk_end_time.timestamp()),
                                      "interval": int(interval)})
        res = send_request(request, tries=-1, delay=20, max_delay=60 * 5,
                           backoff=2).json()["result"]
        aggressor_diff = pd.Series(res["data"], index=pd.to_datetime(res["timestamp"],
                                                                     unit="s"))
        data.append(aggressor_diff)

    data = pd.concat(data).sort_index()
    return data.apply(pd.to_numeric, errors="ignore").astype(float)


def place_order(symbol: str, quantity: float, type_: str = "mkt",
                *,
                limit_price: Optional[float] = None,
                credentials: Credentials):
    side = "sell" if quantity < 0 else "buy"
    params = {
        "orderType": type_,
        "side": side,
        "size": abs(quantity),
        "symbol": symbol,
    }
    order_str = f"{symbol} {quantity}"
    logger.info(f"Placing order: {order_str}")
    if limit_price is not None:
        params["limitPrice"] = limit_price
    # it seems you should also be able to send the params in the request body,
    # but I couldn't get this to work, so we just send them in the query
    # params for now
    request = HttpRequest(method=HttpMethod.POST, base_url=BASE_URL,
                          endpoint="/api/v3/sendorder", params=params)
    request = auth_request(request, credentials)
    try:
        # kraken's API returns 200 status code even if order couldn't be placed,
        # so we must check the send status field of the response ourselves, and
        # raise an exception if it isn't "placed"
        res = send_request(request).json()
        send_status = res["sendStatus"]["status"]
        if send_status != "placed":
            raise ValueError(f"{send_status=}")
    except Exception as e:
        logger.error(f"Error placing order: {order_str}; {e}")
        raise e
