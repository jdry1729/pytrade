import logging
import math
from typing import List, Dict, Optional

import pandas as pd
import requests
import retry

logger = logging.getLogger(__name__)

BASE_URL = "https://tools.morningstar.co.uk/api/rest.svc"

PAGE_SIZE = 1000

PRICE_COLUMNS = [
    "time",
    "price"
]

OHLCV_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume"
]

SCREENER_COLUMNS = ["SecId",
                    "LegalName",
                    "CurrencyId",
                    "Name",
                    "Isin",
                    "Universe",
                    "ExchangeId",
                    "IndustryId",
                    "SectorId",
                    "Ticker",
                    "MarketCountryName"]

SECURITIES_DTYPES = {
    "LegalName": str,
    "CurrencyId": str,
    "Name": str,
    "Isin": str,
    "Universe": str,
    "ExchangeId": str,
    "IndustryId": "string",
    "SectorId": "string",
    "Ticker": str,
}

EXCHANGE_CURRENCY_MAP = {
    "XLON": "GBX",
}


@retry.retry(tries=5, max_delay=10, backoff=1.2)
def _make_request(endpoint: str,
                  params: Dict[str, str]) -> requests.models.Response:
    res = requests.get(f"{BASE_URL}{endpoint}", params=params)
    res.raise_for_status()
    return res


def _get_screener_filter_info(filter_name):
    params = {
        "outputType": "json",
        "version": 1,
        "languageId": "en-GB",
        "filterDataPoints": filter_name
    }
    res = _make_request("/klr5zyak8x/security/screener", params).json()
    data = res["filters"][0][0][filter_name]
    return pd.DataFrame(data)


def get_industries():
    data = _get_screener_filter_info("IndustryId")
    return data.set_index("id")


def get_sectors():
    data = _get_screener_filter_info("SectorId")
    return data.set_index("id")


def get_securities(mic: List[str]):
    params = {
        "outputType": "json",
        "version": 1,
        "languageId": "en-GB",
        "universeIds": "|".join(f"E0EXG${x}" for x in mic)
    }
    results = []
    res = _make_request("/klr5zyak8x/security/screener", params).json()
    total = res["counts"][0]["total"]
    logger.info(
        f"{total} securities found for exchanges: {', '.join(mic)}")
    params["pageSize"] = PAGE_SIZE
    params["securityDataPoints"] = "|".join(SCREENER_COLUMNS)
    num_requests = math.ceil(total / PAGE_SIZE)
    for i in range(num_requests + 1):
        params["page"] = i + 1
        res = _make_request("/klr5zyak8x/security/screener", params).json()
        results.extend(res["rows"])
    results = pd.DataFrame(results)
    results = results.set_index("SecId")
    return results.reindex(columns=list(SECURITIES_DTYPES)).astype(SECURITIES_DTYPES)


# def _get_sec_ids(mic):
#     q = QueryBuilder()
#     lib = pt.get_arctic_lib("morningstar")
#     q = q[q["ExchangeId"] == f"EX$$$${mic}"]
#     securites = lib.read("securities", query_builder=q).data
#     return securites.index


# def _get_complete_sec_ids(tasks, start_date, end_date):
#     # a sec ID is called complete for an interval
#     complete_ids = set()
#     for task in tasks:
#         if task["start_date"] == start_date and task["end_date"] == end_date:
#             complete_ids.update(task["sec_ids"])
#     return complete_ids

# TODO: create generic function to download data from API in chunks
# def _get_morningstar_price_data_tasks(
#         processed_tasks: List[Dict], mic: str, start_date: datetime,
#         end_date: Optional[datetime] = None, settling_period: int = 30,
#         batch_size: int = 100):
#     if end_date is None:
#         end_date = datetime.now()
#
#     sec_ids = _get_sec_ids(mic)
#     currency = EXCHANGE_CURRENCY_MAP[mic]
#     settling_period = timedelta(days=settling_period)
#
#     tasks = []
#     dates = pd.date_range(start_date, end_date,
#                           freq="YS").to_pydatetime().tolist()
#     dates += [end_date]
#     for i in range(len(dates) - 1):
#         chunk_start_date = dates[i]
#         chunk_end_date = dates[i + 1]
#         complete_sec_ids = _get_complete_sec_ids(
#             processed_tasks, chunk_start_date, chunk_end_date)
#         incomplete_sec_ids = sorted(
#             list(sec_ids.difference(complete_sec_ids)))
#         persist = end_date >= chunk_end_date + settling_period
#         if incomplete_sec_ids:
#             for j in range(0, len(incomplete_sec_ids), batch_size):
#                 tasks.append({
#                     "start_date": chunk_start_date.strftime("%Y-%m-%d"),
#                     "end_date": chunk_end_date.strftime("%Y-%m-%d"),
#                     "sec_ids": incomplete_sec_ids[j:j + batch_size],
#                     "currency_id": currency,
#                     "persist": persist
#                 })
#
#     return tasks


def get_price_data(sec_ids: List[str], currency_id: str,
                   start_date: str, end_date: str):
    results = []
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "outputType": "COMPACTJSON",
        "idtype": "Morningstar",
        "frequency": "daily",
        "currencyId": currency_id
    }
    for sec_id in sec_ids:
        params["id"] = sec_id
        res = _make_request("/timeseries_price/t92wz0sj7c", params).json()
        data = pd.DataFrame(res, columns=PRICE_COLUMNS)
        data["security_id"] = sec_id
        results.append(data)
    data = pd.concat(results)
    data["time"] = pd.to_datetime(data["time"], unit="ms")
    return data.set_index(["time", "security_id"]).sort_index()


def get_ohlcv_data(security_ids: List[str], currency_id: str, start_date: str,
                   end_date: Optional[str] = None):
    results = []
    params = {
        "startDate": start_date,
        "endDate": end_date,
        "outputType": "COMPACTJSON",
        "idtype": "Morningstar",
        "frequency": "daily",
        "currencyId": currency_id
    }
    for security_id in security_ids:
        logger.debug(f"Getting OHLCV data for security: {security_id}")
        params["id"] = security_id
        res = _make_request("/timeseries_ohlcv/t92wz0sj7c", params).json()
        data = pd.DataFrame(res, columns=OHLCV_COLUMNS)
        data["security_id"] = security_id
        results.append(data)
    data = pd.concat(results)
    data["time"] = pd.to_datetime(data["time"], unit="ms")
    return data.set_index(["time", "security_id"]).sort_index()
