import json
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Collection, Any

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from pytrade.net.constants import USER_AGENT
from pytrade.net.http import HttpRequest, _send_request, is_http_error, HttpMethod
from pytrade.utils.functions import partial
from pytrade.utils.pandas import empty_df, empty_time_idx
from pytrade.utils.retry import retry

logger = logging.getLogger(__name__)

DATE_COLUMNS = ["transaction_date", "expiration_date"]

INSIDER_TRADES_DTYPES = {
    "transaction_date": "datetime64[ns]",
    "sec_file_number": str,
    "accession_number": str,
    "insider_name": str,
    "insider_title": "object",
    "insider_is_director": bool,
    "insider_is_officer": bool,
    "insider_is_ten_percent_owner": bool,
    "code": str,
    "security_title": str,
    "conversion_exercise_price": float,
    "traded_shares": float,
    "traded_share_price": float,
    "post_transaction_quantity": float,
    "transaction_acquired_disposed_code": str,
    "ownership_nature": str,
    "expiration_date": "datetime64[ns]",
    "is_derivative_transaction": bool,
    "is_holding": bool,
    "filing_url": str,
}

SHORT_INTEREST_DTYPES = {
    "symbol": str,
    "company": str,
    "totalShortInterest": float,
    "daysToCover": float,
    "shortPercentOfFloat": float,
    "shortPriorMo": float,
    "percentChangeMoMo": float,
    "sharesFloat": float,
    "averageDailyVolume": float,
    "sharesOutstanding": float,
    "exchange": str,
    "sector": str,
    "industry": str,
    "exchangeReceiptDate": "datetime64[ns]",
    "settlementDate": "datetime64[ns]",
}

EARNINGS_DTYPES = {
    "currency": "string",
    "cusip": "string",
    "date": "string",
    "date_confirmed": int,
    "eps": "string",
    "eps_est": "string",
    "eps_prior": "string",
    "eps_surprise": "string",
    "eps_surprise_percent": "string",
    "eps_type": "string",
    "exchange": "string",
    "id": "string",
    "importance": int,
    "isin": "string",
    "name": "string",
    "notes": "string",
    "period": "string",
    "period_year": int,
    "revenue": "string",
    "revenue_est": "string",
    "revenue_prior": "string",
    "revenue_surprise": "string",
    "revenue_surprise_percent": "string",
    "revenue_type": "string",
    "ticker": "string",
    "time": "string"
}

RATINGS_DTYPES = {
    "id": "string",
    "ticker": "string",
    "analyst_id": "string",
    "analyst_name": "string",
    "action_company": "string",
    "action_pt": "string",
    "adjusted_pt_current": "string",
    "adjusted_pt_prior": "string",
    "analyst": "string",
    "currency": "string",
    "exchange": "string",
    "importance": int,
    "name": "string",
    "notes": "string",
    "pt_current": "string",
    "pt_prior": "string",
    "rating_current": "string",
    "rating_prior": "string",
    "updated": int,
    "url": "string",
    "url_calendar": "string",
    "url_news": "string",
    "logo": "string",
    "quote": "string"
}

OPTION_ACTIVITY_DTYPES = {
    "aggressor_ind": "string",
    "ask": "string",
    "bid": "string",
    "cost_basis": "string",
    "date": "string",
    "date_expiration": "string",
    "description": "string",
    "description_extended": "string",
    "exchange": "string",
    "execution_estimate": "string",
    "id": "string",
    "midpoint": "string",
    "open_interest": "string",
    "option_activity_type": "string",
    "option_symbol": "string",
    "price": "string",
    "put_call": "string",
    "sentiment": "string",
    "size": "string",
    "strike_price": "string",
    "ticker": "string",
    "trade_count": int,
    "underlying_price": "string",
    "underlying_type": "string",
    "updated": int,
    "volume": "string"
}

INSIDER_TRADES_DATA: Dict[str, Any] = {
    "page": 1,
    "filing_date_preset": "custom",
    "group_by": "filing",
    "sort_by": "last_filing_date",
    "trade_date_preset": "custom",
    "trade_types": [
        "p",
        "s",
        "a",
        "d",
    ]
}


def _get_sec_form_4_entries(filing: Dict):
    # benzinga's schema is same for derivative, non-derivative and holding entries
    non_derivative_holdings = filing["non_derivative"]["holdings"]["entries"]
    non_derivative_holdings = pd.DataFrame(non_derivative_holdings)
    non_derivative_holdings["is_holding"] = True

    non_derivative_trades = filing["non_derivative"]["trades"]
    non_derivative_trade_entries = non_derivative_trades["acquired"]["entries"]
    non_derivative_trade_entries.extend(non_derivative_trades["disposed"]["entries"])
    non_derivative_trade_entries = pd.DataFrame(non_derivative_trade_entries)
    non_derivative_trade_entries["is_holding"] = False
    non_derivative_entries = pd.concat(
        [non_derivative_holdings, non_derivative_trade_entries])

    derivative_entries = pd.DataFrame(filing["derivative"]["entries"])
    derivative_entries["is_holding"] = False

    non_derivative_entries["is_derivative_transaction"] = False
    derivative_entries["is_derivative_transaction"] = True

    entries = pd.concat([derivative_entries, non_derivative_entries])
    entries = entries.rename(columns={"date": "transaction_date",
                                      "title": "security_title"})
    for date_col in DATE_COLUMNS:
        if date_col in entries.columns:
            entries[date_col] = entries[date_col].fillna("").map(lambda x: x[:-1])

    entries["last_filing_date"] = filing["last_filing_date"][:-1]
    entries["sec_file_number"] = filing["sec_file_number"]
    entries["accession_number"] = filing["accession_number"]
    entries["filing_url"] = filing["html_url"]

    if len(filing["insiders"]) > 1:
        logger.warning(f"Multiple reporting persons for filing with accession"
                       f" number: {filing['accession_number']}; ignoring all but"
                       f" first")

    insider = filing["insiders"][0]
    for col in ("name", "is_director", "is_officer", "is_ten_percent_owner"):
        entries[f"insider_{col}"] = insider[col]
    entries["insider_title"] = filing.get("insider_titles_unique", np.nan)

    entries["conversion_exercise_price"] = entries[
        "conversion_exercise_price"].replace(0, np.nan)

    entries["last_filing_date"] = pd.to_datetime(entries["last_filing_date"],
                                                 format="%Y-%m-%dT%H:%M:%S.%f")
    entries = entries.set_index("last_filing_date")
    return entries.reindex(columns=list(INSIDER_TRADES_DTYPES)).astype(
        INSIDER_TRADES_DTYPES)


def get_insider_trades(symbol: str,
                       start_time: Optional[datetime] = datetime(2017, 1, 1),
                       end_time: Optional[datetime] = None,
                       trade_types: Collection[str] = ("P", "S"),
                       page_size: int = 100) -> pd.DataFrame:
    def send_request(data: Dict):
        req = HttpRequest(
            base_url="https://www.benzinga.com",
            endpoint="/sec/insider-trades/api/insider-trades",
            method=HttpMethod.POST,
            json=data,
            headers={
                "User-Agent": USER_AGENT,
            }
        )
        return retry(_send_request, initial_interval=10, e=partial(
            is_http_error, status_code=(429,)), args=(req,)).json()

    if end_time is None:
        end_time = datetime.utcnow()

    entries = []
    data = INSIDER_TRADES_DATA.copy()
    data["company_ticker"] = symbol
    data["page_limit"] = page_size
    data["filing_date_start"] = start_time.strftime("%Y-%m-%d")
    data["filing_date_end"] = end_time.strftime("%Y-%m-%d")
    data["trade_types"] = [x.lower() for x in trade_types]

    res = send_request(data)
    total_pages = res["filings"]["total_pages"]
    for i in range(total_pages):
        data["page"] = i + 1
        res = send_request(data)
        filings = res["filings"]["filings"]
        if len(filings):
            for filing in filings:
                entries.append(_get_sec_form_4_entries(filing))
    if entries:
        return pd.concat(entries).sort_index()
    return pd.DataFrame([], columns=list(INSIDER_TRADES_DTYPES),
                        index=pd.DatetimeIndex([], name="last_filing_date"))


def get_earnings(symbol: str, start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None) -> pd.DataFrame:
    req = HttpRequest(
        base_url="https://www.benzinga.com",
        endpoint=f"/quote/{symbol}/earnings",
        headers={
            "User-Agent": USER_AGENT,
        }
    )
    res = retry(_send_request, initial_interval=10, e=partial(
        is_http_error, status_code=(429,)), args=(req,))
    soup = BeautifulSoup(res.content, parser="html.parser")
    data = json.loads(soup.find("script", {"id": "__NEXT_DATA__"}).text)
    profile = data["props"]["pageProps"]["profile"]
    earnings = pd.DataFrame(profile["earningsSummary"]["earnings"])
    earnings["updated"] = pd.to_datetime(earnings["updated"], unit="s")
    earnings = earnings.set_index("updated").sort_index()
    earnings = earnings.reindex(columns=list(EARNINGS_DTYPES)).astype(EARNINGS_DTYPES)
    if start_time is not None:
        earnings = earnings.loc[start_time:]
    if end_time is not None:
        earnings = earnings.loc[:end_time]
    return earnings


# TODO: more robust way of getting api key?
def get_short_interest_api_key():
    req = HttpRequest(
        base_url="https://www.benzinga.com",
        endpoint="/_next/static/chunks/pages/_app-5e1888c353ff3fd9.js"
    )
    res1 = _send_request(req)
    match = re.search('shortInterestKey:[^"]+"([^"]+)"', res1.text)
    return match.group(1)


def get_analyst_ratings_api_token() -> str:
    req = HttpRequest(
        base_url="https://www.benzinga.com",
        endpoint="/_next/static/chunks/pages/_app-6e206d8aab844f26.js"
    )
    res1 = _send_request(req)
    match = re.search('"benzinga-calendar":\s*{[^}]*token:\s*"([^"]+)"', res1.text)
    return match.group(1)


def get_analyst_ratings(ticker: str, api_token: str,
                        start_time: datetime = datetime(2017, 1, 1),
                        end_time: Optional[datetime] = None) -> pd.DataFrame:
    """
    Gets analyst ratings data.

    Parameters
    ----------
    ticker
    api_token
    start_time
    end_time

    Returns
    -------
    Ratings.
    """
    # TODO: handle pagination?
    if end_time is None:
        end_time = datetime.utcnow()

    start_time_str = start_time.strftime("%Y-%m-%d")
    end_time_str = end_time.strftime("%Y-%m-%d")
    req = HttpRequest(
        base_url="https://api.benzinga.com",
        endpoint="/api/v2.1/calendar/ratings",
        params={
            "token": api_token,
            "pagesize": 1000,
            "parameters[date_from]": start_time_str,
            "parameters[date_to]": end_time_str,
            "parameters[tickers]": ticker,
            # seems the first field below isn't returned, so we repeat id field
            "fields": f"fields=id,date,time,{','.join(RATINGS_DTYPES.keys())}",
        },
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/plain, */*",
        }
    )
    data = retry(_send_request, args=(req,)).json()
    if data:
        data = pd.DataFrame(data["ratings"])
        data["time"] = pd.to_datetime(data["date"] + "T" + data["time"])
        return data.set_index("time").sort_index().reindex(
            columns=list(RATINGS_DTYPES)).astype(RATINGS_DTYPES)
    return empty_df(empty_time_idx(), list(RATINGS_DTYPES)).astype(RATINGS_DTYPES)


def get_option_activity(ticker: str, api_token: str,
                        start_time: datetime = datetime(2017, 1, 1),
                        end_time: Optional[datetime] = None,
                        page_size: int = 1000) -> pd.DataFrame:
    """
    Gets option activity data.

    Parameters
    ----------
    ticker
    api_token
    start_time
    end_time
    page_size

    Returns
    -------
    Option activity.
    """
    if end_time is None:
        end_time = datetime.utcnow()

    page = 0
    start_time_str = start_time.strftime("%Y-%m-%d")
    end_time_str = end_time.strftime("%Y-%m-%d")

    data = []
    while True:
        logger.info(f"Getting page {page} of option activity; {ticker=},"
                    f" {start_time=}, {end_time=}")
        req = HttpRequest(
            base_url="https://api.benzinga.com",
            endpoint="/api/v1/signal/option_activity",
            params={
                "token": api_token,
                "page": page,
                "pagesize": page_size,
                "parameters[date_from]": start_time_str,
                "parameters[date_to]": end_time_str,
                "parameters[tickers]": ticker,
            },
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json, text/plain, */*",
            }
        )
        data_ = retry(_send_request, args=(req,)).json()
        # api returns empty list rather than object if no data
        if not data_:
            break
        data_ = data_["option_activity"]
        data.extend(data_)
        if len(data_) < page_size:
            break
        page += 1

    if data:
        data = pd.DataFrame(data)
        data["time"] = pd.to_datetime(data["date"] + "T" + data["time"])
        return data.set_index("time").sort_index().reindex(
            columns=list(OPTION_ACTIVITY_DTYPES)).astype(OPTION_ACTIVITY_DTYPES)
    return empty_df(empty_time_idx(), list(OPTION_ACTIVITY_DTYPES)).astype(
        OPTION_ACTIVITY_DTYPES)


def get_short_interest(symbol: str,
                       api_key: str,
                       start_time: datetime = datetime(2017, 1, 1),
                       end_time: Optional[datetime] = None) -> pd.DataFrame:
    if end_time is None:
        end_time = datetime.utcnow()

    start_time_str = start_time.strftime("%Y-%m-%d")
    end_time_str = end_time.strftime("%Y-%m-%d")
    req = HttpRequest(
        base_url="https://data-api.benzinga.com",
        endpoint="/rest/v1/shortinterest",
        params={
            "parameters[date_from]": start_time_str,
            "parameters[date_to]": end_time_str,
            "parameters[tickers]": symbol,
            "dateFrom": start_time_str,
            "dateTo": end_time_str,
            "symbols": symbol,
            "apikey": api_key,
            "asOf": start_time_str,
        },
        headers={
            "User-Agent": USER_AGENT,
        }
    )
    data = retry(_send_request, args=(req,)).json()
    # data will be {} if no data in time range
    if data:
        data = pd.DataFrame(data["shortInterestData"][symbol]["data"])
        data["recordDate"] = pd.to_datetime(data["recordDate"])
        data = data.set_index("recordDate")
        # important to select columns in SHORT_INTEREST_DTYPES; that way, an error
        # will be raised if a column is removed from response, and any extra columns
        # returned will be ignored
        data = data[list(SHORT_INTEREST_DTYPES)].astype(SHORT_INTEREST_DTYPES)
        return data.loc[start_time: end_time]
    return pd.DataFrame([], columns=list(SHORT_INTEREST_DTYPES),
                        index=pd.DatetimeIndex([], name="recordDate")).astype(
        SHORT_INTEREST_DTYPES)
