from datetime import datetime

import pandas as pd
from pytrade.net.http import HttpRequest, _send_request
from pytrade.utils.retry import retry

BASE_URL = "https://tranco-list.eu"


def get_list_id(date: datetime, subdomains: bool = False) -> str:
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint="/daily_list_id",
        params={
            "date": date.strftime("%Y-%m-%d"),
            "subdomains": str(subdomains).lower(),
        }
    )
    res = retry(_send_request, args=(req,))
    return res.text


def get_list(date: datetime, subdomains: bool = False) -> pd.Series:
    list_id = get_list_id(date, subdomains)
    data = pd.read_csv(f"{BASE_URL}/download_daily/{list_id}", compression="zip",
                       header=None, names=["rank", "domain"])
    data["time"] = date
    return data.set_index(["time", "rank"])["domain"]
