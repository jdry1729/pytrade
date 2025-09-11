from datetime import datetime

import pandas as pd

from pytrade.net.constants import USER_AGENT
from pytrade.net.http import HttpRequest, HttpMethod, send_request_cffi
from pytrade.utils.retry import retry

BASE_URL = "https://live-api.cryptoquant.com"


def search(query: str):
    params = {
        "page": 1,
        "limit": 10,
    }
    data = {
        "text": query,
        # TODO: what are different types available?
        "type": "ALL",
    }
    req = HttpRequest(base_url=BASE_URL,
                      endpoint="/api/v1/search",
                      method=HttpMethod.POST,
                      json=data,
                      params=params,
                      headers={"User-Agent": USER_AGENT})
    res = retry(send_request_cffi, args=(req,), max_tries=3).json()
    metrics = [{"id": x["id"], "title": x["title"]["en"]} for x in
               res["charts"]["results"]]
    return pd.DataFrame(metrics).set_index("id")


def get_charts(metric_id: str):
    req = HttpRequest(base_url=BASE_URL,
                      endpoint=f"/api/v3/metrics/{metric_id}/charts",
                      headers={"User-Agent": USER_AGENT})
    res = retry(send_request_cffi, args=(req,), max_tries=3).json()
    charts = [{"id": x["id"], "title": x["title"]["en"]} for x in res["charts"]]
    return pd.DataFrame(charts).set_index("id")


def get_chart_data(chart_id: str, window: str, start_time: datetime,
                   end_time: datetime):
    params = {
        "window": window,
        "from": int(start_time.timestamp() * 1000),
        "to": int(end_time.timestamp() * 1000),
        "limit": 70000
    }
    # TODO: need to figure out how to get auth bearer token programatically
    #  without it you don't get latest data
    req = HttpRequest(base_url=BASE_URL,
                      endpoint=f"/api/v3/charts/{chart_id}",
                      params=params,
                      headers={"User-Agent": USER_AGENT})
    res = send_request_cffi(req, raise_for_status=False)
    res = res.json()
    # res = retry(send_request_cffi, args=(req,), max_tries=3).json()
    columns = res["dataKeys"]
    data = res["result"]["data"]
    data = pd.DataFrame(data, columns=columns)
    data["datetime"] = pd.to_datetime(data["datetime"], unit="ms")
    return data.set_index("datetime")
