import logging
import math
from datetime import datetime
from time import sleep
from typing import Optional, Union, Tuple, Dict, Collection, List

import pandas as pd
import urllib3
from pytrade.net.http import HttpRequest, _send_request, HttpMethod, is_http_error
from pytrade.utils.collections import ensure_list, reverse
from pytrade.utils.functions import partial
from pytrade.utils.pandas import empty_df, empty_idx, empty_time_idx
from pytrade.utils.profile import load_profile
from pytrade.utils.retry import retry
from pytrade.utils.time import get_equally_spaced_times, ISO_8601_FORMAT
from requests.exceptions import HTTPError
from tqdm import tqdm

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

FIELDS_MAP = {
    31: "last_price",
    55: "symbol",
    7280: "industry",
    6070: "sec_type",
    6509: "availability",
    84: "bid_price",
    86: "ask_price",
    87: "volume",
    88: "bid_size",
    85: "ask_size",
    7059: "last_size",
}

# must set period so no more than 1000 points returned per request!
BAR_PERIOD_MAP = {
    "5min": "10d",
    "10min": "20d",
    "1h": "120d",
    "1d": "1000d"
}

FIELDS_MAP_REV = reverse(FIELDS_MAP)

POSITIONS_DTYPES = {
    "position": float,
    "avgCost": float,
    "avgPrice": float,
    "currency": str,
    "description": str,
    "isLastToLoq": bool,
    "marketPrice": float,
    "marketValue": float,
    "realizedPnl": float,
    "secType": str,
    "timestamp": int,
    "unrealizedPnl": float,
    "assetClass": str,
    "sector": str,
    "group": str,
    "model": str
}

SECURITY_DEFS_DTYPES = {
    "incrementRules": "object",
    "displayRule": "object",
    "currency": str,
    "time": int,
    "chineseName": str,
    "allExchanges": str,
    "listingExchange": str,
    "countryCode": str,
    "name": str,
    "assetClass": str,
    "expiry": "string",
    "lastTradingDay": "string",
    "group": "string",
    "putOrCall": "string",
    "sector": "string",
    "sectorGroup": "string",
    "strike": float,
    "ticker": str,
    "undConid": int,
    "multiplier": float,
    "type": "string",
    "hasOptions": bool,
    "fullName": str,
    "isUS": bool,
    "isEventContract": bool
}

OHLCV_DTYPES = {
    "o": float,
    "h": float,
    "l": float,
    "c": float,
    "v": float
}


def get_accounts(profile: Optional[str] = None) -> pd.DataFrame:
    profile = load_profile(profile)
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint="/portfolio/accounts",
        verify=False,
    )
    res = _send_request(req)
    return pd.DataFrame(res.json()).set_index("id")


def get_market_data_snapshot(contract_id: Union[int, Collection[int]],
                             fields: Union[str, Collection[str]],
                             pre_flight_delay: int = 2,
                             profile: Optional[str] = None):
    """
    Gets a market data snapshot for contracts.

    Parameters
    ----------
    contract_id
        Contracts to get data for.
    fields
        See here for list: https://www.interactivebrokers.com/campus/ibkr-api-page/
        cpapi-v1/#md-availability.
    pre_flight_delay
        Time to wait after making pre-flight request.
    profile

    Returns
    -------
    Market data snapshot.
    """
    chunk_size = 300
    profile = load_profile(profile)

    contract_ids = ensure_list(contract_id)
    fields = ensure_list(fields)

    fields_int = []
    for field in fields:
        if field in FIELDS_MAP_REV:
            fields_int.append(FIELDS_MAP_REV[field])

    res = []
    params = {"fields": ",".join([str(x) for x in fields_int])}
    # make "pre-flight" request before actual request
    # see: www.interactivebrokers.com/campus/ibkr-api-page/webapi-doc
    for i in range(2):
        for j in range(math.ceil(len(contract_ids) / chunk_size)):
            contract_ids_ = contract_ids[(j * chunk_size):(j + 1) * chunk_size]
            params["conids"] = ",".join(str(x) for x in contract_ids_)
            req = HttpRequest(
                base_url=f"{profile.ib_gateway_uri}/v1/api",
                endpoint="/iserver/marketdata/snapshot",
                params=params,
                verify=False,
            )

            res_ = retry(_send_request, args=(req,), max_tries=2).json()
            if i > 0:
                res.extend(res_)
        if i == 0:
            logger.info(f"Sleeping for {pre_flight_delay}s after making pre-flight"
                        f" requests")
            sleep(pre_flight_delay)

    res = pd.DataFrame(res).set_index("conid")
    res = res.rename(columns={str(k): v for k, v in FIELDS_MAP.items()})
    res["_updated"] = pd.to_datetime(res["_updated"], unit="ms")
    return res[["_updated", "availability"] + [x for x in fields if x in res.columns]]


def close_market_data_streams(profile: Optional[str] = None) -> bool:
    profile = load_profile(profile)
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint="/iserver/marketdata/unsubscribeall",
        method=HttpMethod.GET,
        verify=False,
    )
    res = _send_request(req).json()
    if res["unsubscribed"]:
        return True
    return False


def get_positions(profile: Optional[str] = None) -> pd.DataFrame:
    profile = load_profile(profile)
    params = {
        "direction": "a",
        "sort": "position",

    }
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint=f"/portfolio2/{profile.ib_account_id}/positions",
        verify=False,
        params=params,
    )
    res = _send_request(req).json()
    if res:
        res = pd.DataFrame(res).set_index("conid")
    else:
        res = empty_df(index=empty_idx("conid"), columns=POSITIONS_DTYPES.keys())
    return res.astype(POSITIONS_DTYPES)


def get_portfolio_summary(profile: Optional[str] = None) -> pd.DataFrame:
    profile = load_profile(profile)
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint=f"/portfolio/{profile.ib_account_id}/summary",
        verify=False,
    )
    res = _send_request(req)
    res = pd.DataFrame.from_dict(res.json(), orient="index")
    res["time"] = pd.to_datetime(res["timestamp"], unit="ms")
    return res.drop(columns=["timestamp"])


def get_security(symbol: str, name: bool = False, sec_type: Optional[str] = None,
                 profile: Optional[str] = None):
    profile = load_profile(profile)
    params = {
        "symbol": symbol,
        "name": str(name).lower()
    }
    if sec_type is not None:
        params["secType"] = sec_type
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint="/iserver/secdef/search",
        params=params,
        verify=False,
    )
    res = _send_request(req)
    return res


def get_contract_ids_for_exchange(
        exchange: str, profile: Optional[str] = None) -> pd.Series:
    profile = load_profile(profile)
    params = {
        "exchange": exchange,
    }
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint="/trsrv/all-conids",
        params=params,
        verify=False,
    )
    res = _send_request(req).json()
    res = pd.DataFrame(res).set_index(["exchange", "ticker"])
    return res["conid"].sort_index()


def get_contract_info(contract_id: str, profile: Optional[str] = None) -> pd.Series:
    profile = load_profile(profile)
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint=f"/iserver/contract/{contract_id}/info",
        verify=False,
    )
    res = _send_request(req).json()
    return pd.Series(res)


def get_security_defs(contract_id: Union[int, Collection[int]],
                      profile: Optional[str] = None) -> pd.DataFrame:
    chunk_size = 100
    profile = load_profile(profile)
    # max number of contracts per request is ~100
    contract_ids = [str(x) for x in ensure_list(contract_id)]
    res = []
    for i in tqdm(range(math.ceil(len(contract_ids) / chunk_size))):
        params = {"conids": ",".join(
            contract_ids[(i * chunk_size):(i + 1) * chunk_size])}
        req = HttpRequest(
            base_url=f"{profile.ib_gateway_uri}/v1/api",
            endpoint="/trsrv/secdef",
            params=params,
            verify=False,
        )
        res_ = _send_request(req).json()
        res.extend(res_["secdef"])
    return pd.DataFrame(res).set_index("conid").reindex(
        columns=list(SECURITY_DEFS_DTYPES)).astype(SECURITY_DEFS_DTYPES)


def confirm_warning(warning_id: str, profile: Optional[str] = None) -> Dict:
    profile = load_profile(profile)
    json = {"confirmed": True}
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint=f"/iserver/reply/{warning_id}",
        method=HttpMethod.POST,
        json=json,
        verify=False,
    )
    res = _send_request(req).json()
    return res


def place_order(contract_id: int, quantity: int, order_type: str = "MKT",
                outside_rth: bool = False, price: Optional[float] = None,
                time_in_force: str = "DAY", confirm_warnings: bool = False,
                coid: Optional[str] = None, strategy: Optional[str] = None,
                strategy_params: Optional[Dict] = None, use_adaptive: bool = False,
                profile: Optional[str] = None) -> Union[
    str, Tuple[str, str]]:
    """
    Places an order.

    Parameters
    ----------
    contract_id
    quantity
    order_type
        Order type. E.g., MKT or LMT.
    outside_rth
    price
    time_in_force
    confirm_warnings
    coid
        Optional client order ID.
    strategy
    strategy_params
    use_adaptive
    profile
    """
    profile_name = profile
    profile = load_profile(profile)
    # convert contract_id to int in case numpy int which can't be
    # serialized using json
    order = {
        "conid": int(contract_id),
        "acctId": profile.ib_account_id,
        "side": "SELL" if quantity < 0 else "BUY",
        "quantity": abs(quantity),
        "orderType": order_type,
        "outsideRth": outside_rth,
        "tif": time_in_force,
        "useAdaptive": use_adaptive,
    }
    if price is not None:
        order["price"] = price
    if coid is not None:
        order["cOID"] = coid
    if strategy is not None:
        order["strategy"] = strategy
    if strategy_params is not None:
        order["strategyParameters"] = strategy_params
    logger.info(f"Placing order: {order}")
    json = {"orders": [order]}
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint=f"/iserver/account/{profile.ib_account_id}/orders",
        method=HttpMethod.POST,
        json=json,
        verify=False,
    )
    res = _send_request(req).json()

    while True:
        if "error" in res:
            raise ValueError(f"Error placing order: {res['error']}")

        if len(res) > 1:
            raise ValueError(f"Error placing order; multiple elements in"
                             f" response: {res}")

        res = res[0]
        if "order_id" in res:
            return res["order_id"]

        warning_id = res["id"]
        message = '\n'.join(res['message'])
        if confirm_warnings:
            logger.info(f"Confirming warning; id={warning_id}; message={message}")
            res = confirm_warning(warning_id, profile_name)
        else:
            return warning_id, message


def get_orders(filters: Optional[str] = None, force: bool = True,
               profile: Optional[str] = None) -> pd.DataFrame:
    profile = load_profile(profile)
    params = {
        "force": str(force).lower()
    }
    if filters is not None:
        params["filters"] = filters
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint="/iserver/account/orders",
        params=params,
        verify=False,
    )
    res = _send_request(req).json()
    return pd.DataFrame(res["orders"])


def cancel_order(order_id: str, profile: Optional[str] = None) -> bool:
    profile = load_profile(profile)
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint=f"/iserver/account/{profile.ib_account_id}/order/{order_id}",
        method=HttpMethod.DELETE,
        verify=False,
    )
    res = _send_request(req).json()
    if "order_id" in res:
        return True
    return False


def get_order_status(order_id: str, profile: Optional[str] = None) -> pd.Series:
    profile = load_profile(profile)
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint=f"/iserver/account/order/status/{order_id}",
        verify=False,
    )
    res = _send_request(req).json()
    return pd.Series(res)


def get_performance(account_ids: List[str], period: str = "1D",
                    profile: Optional[str] = None) -> pd.DataFrame:
    profile = load_profile(profile)
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint="/pa/performance",
        json={
            "acctIds": account_ids,
            "period": period,
        },
        verify=False,
        method=HttpMethod.POST,
    )
    res = _send_request(req).json()
    return res


def switch_account(account_id: str, profile: Optional[str] = None) -> bool:
    profile = load_profile(profile)
    json = {"acctId": account_id}
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint="/iserver/account",
        method=HttpMethod.POST,
        json=json,
        verify=False,
    )
    res = _send_request(req).json()
    return res["set"]


def get_auth_status(profile: Optional[str] = None) -> Dict:
    profile = load_profile(profile)
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint="/iserver/auth/status",
        method=HttpMethod.POST,
        verify=False,
    )
    res = _send_request(req).json()
    return res


def get_trades(days: Optional[int] = None,
               profile: Optional[str] = None) -> pd.DataFrame:
    """
    Gets trades for an account.

    Parameters
    ----------
    days
    profile

    Returns
    -------
    Trades.

    Notes
    -----
    You must call switch_account with the ID of the account which you're
    trying to get trades for before calling this function.
    """
    params = {}
    if days is not None:
        params["days"] = days
    profile = load_profile(profile)
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint="/iserver/account/trades",
        params=params,
        method=HttpMethod.GET,
        verify=False,
    )
    res = _send_request(req).json()
    res = pd.DataFrame(res)
    res["quantity"] = res["size"].where(res["side"] == "B", -res["size"])
    res["time"] = pd.to_datetime(res["trade_time_r"], unit="ms")
    return res.set_index(["time", "symbol"]).sort_index()[
        ["price", "commission", "net_amount", "quantity"]].astype(float)


def wait_for_auth(interval: int = 10, timeout: Optional[int] = 60 * 60,
                  profile: Optional[str] = None) -> None:
    total_time = 0
    while True:
        try:
            # session must be authenticated to place orders
            status = get_auth_status(profile=profile)
            if status["authenticated"]:
                logger.info("CP gateway authenticated")
                return
            logger.info("CP gateway not authenticated")
        except Exception:
            pass
        sleep(interval)
        total_time += interval
        if timeout is not None and total_time > timeout:
            break
    raise ValueError("Error waiting for CP gateway authentication; timeout")


def reauthenticate(profile: Optional[str] = None) -> bool:
    """
    Can be used to establish a brokerage session with Interactive Brokers (which is
    needed to access /iserver endpoints).
    """
    profile = load_profile(profile)
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint="/iserver/reauthenticate",
        method=HttpMethod.POST,
        verify=False,
    )
    res = _send_request(req).json()
    if res["message"] == "triggered":
        return True
    return False


def get_algos_for_contract(contract_id: int, profile: Optional[str] = None):
    profile = load_profile(profile)
    req = HttpRequest(
        base_url=f"{profile.ib_gateway_uri}/v1/api",
        endpoint=f"/iserver/contract/{contract_id}/algos",
        params={
            "addDescription": "1",
            "addParams": "1",
        },
        method=HttpMethod.GET,
        verify=False,
    )
    res = _send_request(req).json()
    return res


def get_ohlcv_data(
        conid: int,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        freq: str = "1d",
        show_progress: bool = False,
        profile: Optional[str] = None
) -> pd.DataFrame:
    """
    Gets OHLCV data for a contract.

    Parameters
    ----------
    conid
    start_time
    end_time
    freq
        May be 1min, 5min, 10min, 1h or 1d.
    show_progress
    profile

    Returns
    -------
    Data.
    """
    profile = load_profile(profile)
    if end_time is None:
        end_time = datetime.utcnow()

    period = BAR_PERIOD_MAP[freq]
    times = get_equally_spaced_times(start_time, end_time, period=pd.Timedelta(period))

    if end_time not in times:
        times.append(end_time)

    data = []
    for i, time in enumerate(tqdm(times[1:], disable=not show_progress)):
        req = HttpRequest(
            base_url=f"{profile.ib_gateway_uri}/v1/api",
            endpoint="/iserver/marketdata/history",
            verify=False,
            params={
                "conid": str(conid),
                "period": period,
                "bar": freq,
                # oddly, data is fetched up to start time exclusive
                "startTime": time.strftime("%Y%m%d-%H:%M:%S"),
            },
        )
        try:
            res = retry(_send_request, args=(req,),
                        e=partial(is_http_error, status_code=(400, 429, 503)),
                        max_tries=10)
        except HTTPError as e:
            if is_http_error(e, 500) and e.response.json().get("error") == "No data.":
                logger.debug(f"No data found for contract {conid} from"
                             f" {times[i].strftime(ISO_8601_FORMAT)} to"
                             f" {times[i + 1].strftime(ISO_8601_FORMAT)}")
                continue
            raise e
        else:
            data.extend(res.json()["data"])

    # first bar for new asset doesn't have t (i.e., time) field
    data = [x for x in data if "t" in x]
    if data:
        data = pd.DataFrame(data)
        data["t"] = pd.to_datetime(data["t"], unit="ms")
        data = data.set_index("t").sort_index()
        data = data.loc[~data.index.duplicated(keep="last")]
        return data.loc[start_time:end_time].reindex(
            columns=list(OHLCV_DTYPES.keys())).astype(OHLCV_DTYPES)
    return empty_df(index=empty_time_idx("t"),
                    columns=list(OHLCV_DTYPES.keys())).astype(OHLCV_DTYPES)
