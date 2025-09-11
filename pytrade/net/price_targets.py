from datetime import datetime
from io import StringIO
from typing import Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pytrade.net.http import HttpRequest, _send_request, is_http_error
from pytrade.utils.functions import partial
from pytrade.utils.retry import retry

PRICE_TARGETS_DTYPES = {
    "Action": "object",
    "Rating": "object",
    "Price Target": "object",
    "Prev Price Target": float,
    "New Price Target": float
}


def get_price_target_history(symbol: str, exchange: str,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None):
    """

    Parameters
    ----------
    symbol
        Symbol to get data for.
    exchange
        Exchange the symbol is listed on.
    start_time
    end_time

    Returns
    -------

    """
    req = HttpRequest(
        base_url="https://www.pricetargets.com",
        endpoint=f"/{exchange}/{symbol}/"
    )
    res = retry(_send_request, initial_interval=10, e=partial(
        is_http_error, status_code=(429,)), args=(req,))
    soup = BeautifulSoup(res.content, "html.parser")
    upgrades_div = soup.find("div", {"id": "upgrades-and-downgrades"})
    if upgrades_div is None:
        raise ValueError("Error getting price target history; upgrades/ downgrades"
                         " table not found")
    upgrades_table = str(upgrades_div.find("table"))
    # use StringIO below to avoid pandas future warning
    data = pd.read_html(StringIO(upgrades_table))[0]
    data["Date"] = pd.to_datetime(data["Date"])
    # TODO: handle case where brokerage issues mutliple targets on same day?
    data = data.set_index(["Date", "Brokerage"]).sort_index()
    data["Price Target"] = data["Price Target"].str.replace(r'[^0-9.➝]', '', regex=True)
    data[["Prev Price Target", "New Price Target"]] = data[
        "Price Target"].str.split("➝", expand=True).replace([None], value=np.nan)
    data["Prev Price Target"] = pd.to_numeric(data["Prev Price Target"])
    data["New Price Target"] = pd.to_numeric(data["New Price Target"])
    data["New Price Target"] = data["New Price Target"].fillna(
        data["Prev Price Target"])
    data = data.drop(columns=["Details"])
    data = data.reindex(columns=list(PRICE_TARGETS_DTYPES)).astype(
        PRICE_TARGETS_DTYPES)
    if start_time is not None:
        data = data.loc[start_time:]
    if end_time is not None:
        data = data.loc[:end_time]
    return data
