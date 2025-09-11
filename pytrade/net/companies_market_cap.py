import json
import re

import jsmin
import pandas as pd
from bs4 import BeautifulSoup
from pytrade.net.constants import USER_AGENT
from pytrade.net.http import HttpMethod, HttpRequest, _send_request
from pytrade.utils.retry import retry

BASE_URL = "https://companiesmarketcap.com"


def get_company_path(symbol: str) -> str:
    req = HttpRequest(
        BASE_URL,
        endpoint="/search.do",
        method=HttpMethod.POST,
        data={"query": symbol},
        headers={"User-Agent": USER_AGENT},
    )
    res = retry(_send_request, args=(req,)).json()
    for x in res:
        if x["identifier"] == symbol:
            return x["url"]
    raise ValueError(f"Error getting company URL for ticker: {symbol}")


def get_market_cap_history(company_path: str) -> pd.DataFrame:
    req = HttpRequest(
        BASE_URL,
        endpoint=f"/{company_path}/marketcap",
        method=HttpMethod.GET,
        headers={"User-Agent": USER_AGENT},
    )
    res = retry(_send_request, args=(req,))
    soup = BeautifulSoup(res.content, "html.parser")
    data = soup.find("script", string=lambda x: x and "data =" in x)
    script_content = jsmin.jsmin(data.text)
    match = re.search(r"data=(\[.*\]);", script_content, re.DOTALL)
    data = pd.DataFrame(json.loads(match.group(1)))
    data["d"] = pd.to_datetime(data["d"], unit="s")
    return data.set_index("d")["m"]
