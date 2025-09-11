from datetime import datetime
from typing import Optional, Collection

import pandas as pd
from pytrade.net.constants import USER_AGENT_2
from pytrade.net.http import HttpMethod, HttpRequest, _send_request
from pytrade.utils.retry import retry


def get_rankings(token: str, *, app_id: Optional[int] = None,
                 store_app_id: Optional[str] = None, start_time: datetime,
                 end_time: Optional[datetime] = None, store: int = 5,
                 category: Optional[int] = None, freq: str = "day",
                 countries: Collection[str] = ("WW",),
                 retry_initial_interval: int = 120,
                 retry_max_interval: Optional[int] = 600,
                 max_tries: int = 10,
                 retry_multiplier: int = 2) -> pd.Series:
    if app_id is None and store_app_id is None:
        raise ValueError("Error getting rankings; at least one of app ID and store"
                         " app ID must be specified")
    if end_time is None:
        end_time = datetime.utcnow()

    data = {
        "store": store,
        "aggregation": freq,
        "countries": list(countries),
        "dateStart": start_time.strftime("%Y-%m-%d"),
        "dateEnd": end_time.strftime("%Y-%m-%d"),
    }

    if app_id is not None:
        endpoint = "/api/v2/charts/united-applications/multi"
        data["id"] = app_id
    else:
        endpoint = "/api/v2/charts/applications/multi"
        data["store_application_id"] = store_app_id

    if category is not None:
        data["category"] = category

    req = HttpRequest(
        base_url="https://appmagic.rocks",
        endpoint=endpoint,
        headers={
            "authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT_2,
        },
        method=HttpMethod.POST,
        json={"requests": [data]},
    )
    res = retry(_send_request, args=(req,),
                initial_interval=retry_initial_interval,
                max_interval=retry_max_interval,
                max_tries=max_tries,
                multiplier=retry_multiplier).json()[0]
    data = pd.DataFrame(res["top_free"]["points"], columns=["time", "ranking"])
    data["time"] = pd.to_datetime(data["time"], unit="ms")
    return data.set_index("time")["ranking"]
