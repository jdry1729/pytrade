from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Optional

import pandas as pd
from pytrade.net.http import HttpRequest, HttpMethod, _send_request
from pytrade.utils.retry import retry

logger = logging.getLogger(__name__)

BASE_URL = "https://www.deribit.com"
HISTORY_BASE_URL = "https://history.deribit.com"

INSTRUMENTS_COLUMNS = [
    "tick_size",
    "taker_commission",
    "settlement_period",
    "settlement_currency",
    "rfq",
    "quote_currency",
    "price_index",
    "min_trade_amount",
    "max_liquidation_commission",
    "max_leverage",
    "maker_commission",
    "kind",
    "is_active",
    "instrument_name",
    "instrument_id",
    "instrument_type",
    "expiration_timestamp",
    "creation_timestamp",
    "counter_currency",
    "contract_size",
    "block_trade_tick_size",
    "block_trade_min_trade_amount",
    "block_trade_commission",
    "base_currency",
    "strike",
    "option_type",
]

TRADES_COLUMNS = [
    "trade_seq",
    "trade_id",
    "timestamp",
    "tick_direction",
    "price",
    "mark_price",
    "instrument_name",
    "index_price",
    "direction",
    "amount",
]


@dataclass
class Credentials:
    client_id: str
    client_secret: str


def auth_request(request: HttpRequest, credentials: Credentials):
    headers = {}
    if credentials.client_id is not None:
        auth = f"{credentials.client_id}:{credentials.client_secret}"
        headers["Authorization"] = f"Basic {base64.b64encode(auth, 'utf-8')}"
    request = replace(request, headers=headers)
    return request


def get_instruments(currency: Optional[str] = None, kind: Optional[str] = None,
                    expired: bool = False):
    base_url = BASE_URL
    params = {"expired": "false"}
    if expired:
        if currency is None:
            raise ValueError("Error getting instruments; currency must be specified if"
                             " expired is True")
        params["expired"] = "true"
        # TODO: HISTORY_BASE_URL doesn't provide data for options which have very
        #  recently expired.. perhaps need to get data from both base urls if
        #  expired is true?
        base_url = HISTORY_BASE_URL
    if currency is not None:
        params["currency"] = currency
    if kind is not None:
        params["kind"] = kind
    request = HttpRequest(method=HttpMethod.GET, base_url=base_url,
                          endpoint="/api/v2/public/get_instruments",
                          params=params)
    data = _send_request(request).json()
    instruments = pd.DataFrame(data["result"], columns=INSTRUMENTS_COLUMNS)
    instruments["creation_time"] = pd.to_datetime(
        instruments["creation_timestamp"], unit="ms")
    # deribit sets expiration time of perpetuals to 3000-01-01; we specify
    # errors="coerce" below so this gets converted to NaT
    instruments["expiration_time"] = pd.to_datetime(
        instruments["expiration_timestamp"], unit="ms", errors="coerce")
    instruments = instruments.drop(
        columns=["creation_timestamp", "expiration_timestamp"])
    return instruments.set_index("instrument_id")


def get_trades(instrument_name: str, start_time: datetime,
               end_time: Optional[datetime] = None):
    # historical trades endpoint is updated 5s after the trade actually occurred, so
    # this function might not be suitable for time-sensitive applications
    # see: https://twitter.com/DeribitExchange/status/1570135181033607168
    if end_time is None:
        end_time = datetime.utcnow()
    params = {
        "instrument_name": instrument_name,
        "start_timestamp": int(start_time.timestamp() * 1000),
        "end_timestamp": int(end_time.timestamp() * 1000),
        "sorting": "asc",
        "count": 10000,
    }
    request = HttpRequest(method=HttpMethod.GET, base_url=HISTORY_BASE_URL,
                          endpoint="/api/v2/public/get_last_trades_by_instrument",
                          params=params)

    trades = []
    while True:
        res = retry(_send_request, max_tries=10, args=(request,))
        trades_ = res.json()["result"]["trades"]
        if len(trades_) == 0:
            break
        trades.extend(trades_)
        # TODO: use trade_seq instead?
        params["start_timestamp"] = trades_[-1]["timestamp"] + 1
        request = replace(request, params=params)

    trades = pd.DataFrame(trades, columns=TRADES_COLUMNS)
    trades["time"] = pd.to_datetime(trades["timestamp"], unit="ms")
    return trades.set_index("trade_id").sort_values("trade_seq")
