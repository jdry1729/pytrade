import pandas as pd
from pytrade.net.http import HttpRequest, _send_request, is_http_error
from pytrade.utils.functions import partial
from pytrade.utils.retry import retry

BASE_URL = "https://api.stockanalysis.com"


def get_market_cap_history(symbol: str) -> pd.Series:
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint=f"/api/symbol/s/{symbol.lower()}/marketcap",
        params={
            "t": "price"
        }
    )
    res = retry(_send_request, e=partial(
        is_http_error, status_code=(429,)), args=(req,)).json()
    data = pd.DataFrame(res["data"], columns=["time", "market_cap"])
    data["time"] = pd.to_datetime(data["time"], unit="ms")
    return data.set_index("time")["market_cap"]
