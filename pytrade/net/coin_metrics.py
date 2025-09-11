import logging
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd

from pytrade.net.http import HttpMethod, HttpRequest, send_request

BASE_URL = "https://community-api.coinmetrics.io/v4"

logger = logging.getLogger(__name__)


def get_metric_details(metrics: Optional[Iterable[str]] = None):
    details = []
    next_page_token = None
    params = {
        "page_size": 10000,
        "paging_from": "start",
    }
    if metrics is not None:
        params["metrics"] = ",".join(metrics),
    while True:
        if next_page_token is not None:
            params["next_page_token"] = next_page_token
        request = HttpRequest(
            BASE_URL,
            "/reference-data/asset-metrics",
            method=HttpMethod.GET,
            params=params,
        )
        res = send_request(request).json()

        details.extend(res["data"])
        if "next_page_token" in res:
            next_page_token = res["next_page_token"]
        else:
            break
    return pd.DataFrame(details).set_index("metric")


def get_available_metrics(assets):
    metrics = []
    next_page_token = None
    params = {
        "assets": ",".join(assets),
        "page_size": 10000,
        "paging_from": "start",
    }
    while True:
        if next_page_token is not None:
            params["next_page_token"] = next_page_token
        request = HttpRequest(
            BASE_URL,
            "/catalog-v2/asset-metrics",
            method=HttpMethod.GET,
            params=params,
        )
        res = send_request(request).json()

        data = res["data"]
        for asset_data in data:
            for asset_metric in asset_data["metrics"]:
                for freq in asset_metric["frequencies"]:
                    metrics.append(
                        {
                            "asset": asset_data["asset"],
                            "metric": asset_metric["metric"],
                            "frequency": freq["frequency"],
                            "min_time": freq["min_time"],
                            "max_time": freq["max_time"],
                            "community": freq["community"],
                        }
                    )
        if "next_page_token" in res:
            next_page_token = res["next_page_token"]
        else:
            break
    return pd.DataFrame(metrics)


def get_timeseries(assets: Iterable[str], metrics: Iterable[str], start_time: datetime,
                   end_time: datetime):
    data = []
    next_page_token = None
    params = {
        "metrics": metrics,
        "assets": ",".join(assets),
        "start_time": start_time.strftime("%Y-%m-%d"),
        "end_time": end_time.strftime("%Y-%m-%d"),
        "page_size": 10000,
        "paging_from": "start",
        "sort": "time",
    }
    while True:
        if next_page_token is not None:
            params["next_page_token"] = next_page_token
        request = HttpRequest(
            BASE_URL,
            "/timeseries/asset-metrics",
            method=HttpMethod.GET,
            params=params,
        )
        # community version of API has rate limit of 10 requests per 6 seconds
        res = send_request(request, tries=10, delay=5).json()
        data.extend(res["data"])
        if "next_page_token" in res:
            next_page_token = res["next_page_token"]
        else:
            break
    data = pd.DataFrame(data)
    data["time"] = pd.to_datetime(data["time"]).dt.tz_localize(None)
    data = data.set_index(["time", "asset"])
    data = data.apply(pd.to_numeric, errors="ignore").astype(float)
    return data
