import logging
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional

import pandas as pd
from pytrade.net.constants import USER_AGENT_2
from pytrade.net.http import HttpMethod, HttpRequest, _send_request
from pytrade.utils.pandas import empty_df, empty_time_idx
from pytrade.utils.retry import retry
from pytrade.utils.time import get_equally_spaced_times
from tqdm import tqdm

logger = logging.getLogger(__name__)

SHORT_INTEREST_DTYPES = {
    "stockSplitFlag": "string",
    "previousShortPositionQuantity": int,
    "averageDailyVolumeQuantity": int,
    "issueName": "string",
    "currentShortPositionQuantity": int,
    "changePreviousNumber": int,
    "accountingYearMonthNumber": int,
    "marketClassCode": "string",
    "symbolCode": "string",
    "daysToCoverQuantity": float,
    "issuerServicesGroupExchangeCode": "string",
    "revisionFlag": "string",
    "changePercent": float,
}

SHORT_VOLUME_DTYPES = {
    "ShortVolume": int,
    "ShortExemptVolume": int,
    "TotalVolume": int,
    "Market": str,
}


def get_access_token(client_id: str, secret: str):
    req = HttpRequest(
        base_url="https://ews.fip.finra.org",
        endpoint="/fip/rest/ews/oauth2/access_token",
        params={
            "grant_type": "client_credentials",
        },
        method=HttpMethod.POST,
        auth=(client_id, secret),
    )
    res = _send_request(req)
    return res.json()["access_token"]


def get_short_interest(symbol: str, access_token: str, start_time: datetime,
                       end_time: datetime) -> pd.DataFrame:
    """
    Gets short interest.

    Parameters
    ----------
    symbol
    access_token
    start_time
        Start time (inclusive).
    end_time
        End time (inclusive).

    Returns
    -------
    Short interest.
    """
    times = get_equally_spaced_times(start_time, end_time, timedelta(days=364))
    if end_time not in times:
        times.append(end_time)
    data = []
    for i in tqdm(range(1, len(times))):
        req = HttpRequest(
            base_url="https://api.finra.org",
            endpoint="/data/group/otcmarket/name/consolidatedShortInterest",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            method=HttpMethod.POST,
            json={
                "limit": 1000,
                "compareFilters": [
                    {
                        "fieldName": "symbolCode",
                        "fieldValue": symbol,
                        "compareType": "EQUAL",
                    }
                ],
                "dateRangeFilters": [
                    {
                        "startDate": times[i - 1].strftime("%Y-%m-%d"),
                        "endDate": times[i].strftime("%Y-%m-%d"),
                        "fieldName": "settlementDate",
                    }
                ],
            },
        )
        res = retry(_send_request, args=(req,))
        # status code will be 204 if no data available
        if res.status_code != 204:
            data.extend(res.json())
    if data:
        data = pd.DataFrame(data)
        data["settlementDate"] = pd.to_datetime(data["settlementDate"])
        return data.set_index("settlementDate").sort_index().reindex(
            columns=list(SHORT_INTEREST_DTYPES.keys())
        ).astype(SHORT_INTEREST_DTYPES)
    return empty_df(index=empty_time_idx("settlementDate"),
                    columns=list(SHORT_INTEREST_DTYPES.keys())).astype(
        SHORT_INTEREST_DTYPES)


def get_short_volume(start_time: datetime, end_time: Optional[datetime] = None):
    if end_time is None:
        end_time = datetime.utcnow()

    data = []
    for time in tqdm(pd.date_range(start_time, end_time, freq="B")):
        time_str = time.strftime("%Y%m%d")
        req = HttpRequest(
            base_url="https://cdn.finra.org",
            endpoint=f"/equity/regsho/daily/CNMSshvol{time_str}.txt",
            headers={"User-Agent": USER_AGENT_2},
        )
        try:
            res = retry(_send_request, args=(req,), max_tries=2)
            data_ = pd.read_csv(
                StringIO(res.text),
                sep="|",
                skipfooter=1,
                engine="python",
            )
            data_["Date"] = pd.to_datetime(data_["Date"], format="%Y%m%d")
            data.append(data_.set_index(["Date", "Symbol"]))
        except Exception:
            logger.warning(f"Error downloading data for: {time_str}")
    if data:
        return (
            pd.concat(data)
            .reindex(columns=list(SHORT_VOLUME_DTYPES.keys()))
            .astype(SHORT_VOLUME_DTYPES)
        )
    return empty_df(
        index=pd.MultiIndex.from_product(
            [pd.DatetimeIndex([]), pd.Index([])], names=["Date", "Symbol"]
        ),
        columns=list(SHORT_VOLUME_DTYPES.keys()),
    ).astype(SHORT_VOLUME_DTYPES)
