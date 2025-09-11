import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from pytrade.net.constants import USER_AGENT
from pytrade.net.http import HttpRequest, _send_request
from pytrade.utils.retry import retry

logger = logging.getLogger(__name__)


def get_page_views(page: str, start_time: datetime,
                   end_time: Optional[datetime], agent: str = "user"):
    if end_time is None:
        end_time = datetime.utcnow()

    start_date_str = start_time.strftime("%Y%m%d")
    end_date_str = end_time.strftime("%Y%m%d")
    req = HttpRequest(
        base_url="https://wikimedia.org",
        endpoint=(f"/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/"
                  f"all-access/{agent}/{page}/daily/{start_date_str}/{end_date_str}"),
        headers={"User-Agent": USER_AGENT}
    )
    res = retry(_send_request, args=(req,))

    data = pd.DataFrame(res.json()["items"])
    data["time"] = pd.to_datetime(data["timestamp"], format="%Y%m%d%H")
    return data.set_index("time")["views"].sort_index()
