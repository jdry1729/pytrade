from datetime import datetime
from typing import Optional

import requests

BASE_URL = "https://api.coinmarketcap.com/data-api/v3"


def get_historical_listings(date: datetime, limit: Optional[int] = None):
    url = f"{BASE_URL}/cryptocurrency/listings/historical"
    params = {"date": date.strftime("%Y-%m-%d")}
    if limit is not None:
        params["limit"] = limit
    res = requests.get(url, params=params)
    res.raise_for_status()
    return res.json()["data"]
