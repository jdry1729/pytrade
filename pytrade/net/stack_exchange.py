from datetime import timedelta, datetime
from typing import Optional

import pandas as pd
from pytrade.net.http import send_request, HttpRequest, HttpMethod
from tqdm import tqdm

BASE_URL = "http://api.stackexchange.com"

# filters below just get the total number of results returned
FILTER_MAP = {
    "users": "!40CXQQCnzNTLhV4ep",
    "posts": "!CdjJUYVXONv9YMyqFq-lv",
    "answers": "!HUWWJ)6LYhiHX71CO",
    "comments": "!HUWWJ)6LYhiHX71CO",
}


def get_sites():
    request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                          endpoint="/2.3/sites", params={"pagesize": 10000})
    res = send_request(request).json()
    sites = [
        {"api_site_parameter": x["api_site_parameter"], "name": x["name"],
         "launch_date": x.get("launch_date", None), "site_state": x["site_state"],
         "open_beta_date": x.get("open_beta_date", None),
         "closed_beta_date": x.get("closed_beta_date", None)} for x
        in res["items"]]
    sites = pd.DataFrame(sites).set_index("api_site_parameter")
    for col in ["launch_date", "open_beta_date", "closed_beta_date"]:
        sites[col] = pd.to_datetime(sites[col], unit="s")
    return sites


def _count_objs(site: str, obj_type: str, start_time: datetime,
                end_time: datetime, key: Optional[str] = None) -> pd.Series:
    count = {}
    date_range = pd.date_range(start_time, end_time)
    filter_ = FILTER_MAP[obj_type]
    for i in tqdm(range(len(date_range))):
        # it's no faster to request data for 1 day period than 10 year period,
        # so we just request number of
        end_time_ = date_range[i]
        params = {
            "fromdate": int(datetime(2000, 1, 1).timestamp()),
            "todate": int((end_time_ - timedelta(seconds=1)).timestamp()),
            "site": site,
            "filter": filter_,
        }
        if key is not None:
            params["key"] = key
        request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                              endpoint=f"/2.3/{obj_type}", params=params)
        res = send_request(request, tries=10)
        count[end_time_.to_pydatetime()] = res.json()["total"]
    return pd.Series(count)


def get_num_users(site, start_time, end_time,
                  key: Optional[str] = None) -> pd.Series:
    return _count_objs(site, "users", start_time, end_time, key)


def get_num_posts(site, start_time, end_time,
                  key: Optional[str] = None) -> pd.Series:
    """
    A post refers to a question or an answer.
    """
    return _count_objs(site, "posts", start_time, end_time, key)


def get_num_answers(site, start_time, end_time,
                    key: Optional[str] = None) -> pd.Series:
    return _count_objs(site, "answers", start_time, end_time, key)


def get_num_comments(site, start_time, end_time,
                     key: Optional[str] = None) -> pd.Series:
    return _count_objs(site, "comments", start_time, end_time, key)
