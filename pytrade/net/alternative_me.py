import pandas as pd

from pytrade.net.constants import USER_AGENT
from pytrade.net.http import HttpRequest, HttpMethod, _send_request, send_request_cffi
from pytrade.utils.retry import retry

BASE_URL = "https://alternative.me"


def get_fear_and_greed_index(days: int = 1000):
    endpoint = "/api/crypto/fear-and-greed-index/history"
    req = HttpRequest(base_url=BASE_URL,
                      endpoint=endpoint,
                      method=HttpMethod.POST,
                      json={"days": days},
                      headers={"User-Agent": USER_AGENT})
    res = retry(send_request_cffi, args=(req,)).json()["data"]
    data = res["datasets"][0]["data"]
    return pd.Series(data, index=pd.to_datetime(res["labels"]))
