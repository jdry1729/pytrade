import logging
from datetime import datetime
from typing import Iterable

import pandas as pd
from pytrade.net.http import HttpRequest, HttpMethod, send_request_cffi
from pytrade.utils.pandas import stack
from pytrade.utils.retry import retry

BASE_URL = "https://api-prod.etf.com/private/apps/fundflows"

logger = logging.getLogger(__name__)


def get_fund_flows(tickers: Iterable[str], start_time: datetime,
                   end_time: datetime) -> pd.DataFrame:
    data = {}
    for ticker in tickers:
        try:
            req = HttpRequest(base_url=BASE_URL,
                              method=HttpMethod.GET,
                              endpoint=f"/{ticker}/charts",
                              params={
                                  "startDate": start_time.strftime("%Y%m%d"),
                                  "endDate": end_time.strftime("%Y%m%d"),
                              })
            res = retry(send_request_cffi, max_tries=3, request=req,
                        raise_for_status=True).json()
            data_ = pd.DataFrame(res["data"]["results"]["data"])
            data_["time"] = pd.to_datetime(data_["asOf"])
            data[ticker] = data_.set_index("time")["value"]
        except Exception as e:
            logger.info(f"Error getting fund flows for: {ticker}; {e}")
    return stack(data, names="ticker")
