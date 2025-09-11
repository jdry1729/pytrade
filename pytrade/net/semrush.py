import json
from typing import Iterable

import pandas as pd
from pytrade.net.webdriver import get_performance_logs, get_fetch_xhr_requests
from pytrade.utils.time import sleep
from selenium import webdriver

TRAFFIC_RPC_METHOD = "organic.OverviewTrend"


def _find_overview_trend_req(reqs: Iterable, freq: str):
    for req in reqs:
        data = json.loads(req["params"]["request"]["postData"])
        if isinstance(data, dict) and data["method"] == TRAFFIC_RPC_METHOD:
            if data["params"]["args"]["dateType"] == freq:
                return req
    raise ValueError("Error finding req")


def get_traffic(driver: webdriver.Chrome, domain: str,
                freq: str = "monthly", delay: int = 5) -> pd.DataFrame:
    """
    Notes
    -----
    To avoid Semrush login page (which has a captcha), you have to use a Chrome
    profile which has already completed the captcha challenge.

    To do this, you could copy your working user data dir (which you use when
    manually using Chrome) from ~/.config/google-chrome to /tmp/google-chrome
    and then instantiate the webdriver as below. I'm not sure why you have to
    copy the chrome profile into tmp.. but it works!

    driver = chromedriver(
        headless=False,
        binary_location="/usr/bin/chromium-browser",
        user_data_dir="/tmp/google-chrome/",
        profile_dir="Default",
        logging_prefs={"performance": "ALL"},
    )
    """
    get_performance_logs(driver)
    driver.get(f"https://www.semrush.com/analytics/overview/?q={domain}"
               f"&protocol=https&searchType=domain")

    sleep(delay)
    logs = get_performance_logs(driver)
    reqs = get_fetch_xhr_requests(logs, endpoint="/dpa/rpc")

    req = _find_overview_trend_req(reqs, freq)

    request_id = req["params"]["requestId"]
    data = driver.execute_cdp_cmd("Network.getResponseBody",
                                  {"requestId": request_id})

    body = json.loads(data["body"])
    if "error" in body:
        raise ValueError(f"Error getting traffic; {body['error']['message']}")

    data = pd.DataFrame(json.loads(data["body"])["result"])
    data["date"] = pd.to_datetime(data["date"])
    return data.set_index("date").drop(
        columns=["organicPositionsTrend", "adwordsPositionsTrend"])
