import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from pytrade.net.http import _send_request, HttpRequest, is_http_error
from pytrade.utils.functions import partial
from pytrade.utils.retry import retry

BASE_URL = "https://graph.facebook.com/v20.0"

ENDPOINT = "/me/accounts"

MEDIA_COLUMNS = ["comments_count", "like_count", "timestamp", "id"]

logger = logging.getLogger(__name__)


# facebook/ instagram rate limits seem to reset after an hour

def get_long_lived_user_access_token(app_id, app_secret, short_lived_user_access_token):
    """
    Gets long-lived user access token. Typically expires after 60 days.

    Parameters
    ----------
    app_id
        ID of app to use to generate the token. Can be found here:
        https://developers.facebook.com/apps/?show_reminder=true.
    app_secret
        secret of app to use. Can be found here:
        https://developers.facebook.com/apps/?show_reminder=true.
    short_lived_user_access_token
        Short-lived user access token. Can be generated here
        https://developers.facebook.com/tools/explorer.

    Returns
    -------
    Long-lived user-access token.
    """
    params = {
        "grant_type": "fb_exchange_token",
        "client_id": app_id,
        "client_secret": app_secret,
        "fb_exchange_token": short_lived_user_access_token
    }
    req = HttpRequest(base_url=BASE_URL, endpoint="/oauth/access_token", params=params)
    res = _send_request(req)
    return res.json()["access_token"]


def get_long_lived_page_access_token(app_scoped_user_id: int,
                                     long_lived_user_access_token: str):
    """
    Gets a long lived page access token from an app-scoped user ID and a long-lived
    user access token. Long lived page access tokens don't expire except in special
    circumstances.

    Parameters
    ----------
    app_scoped_user_id
        App-scoped user ID. Can be found by going to this endpoint
        https://developers.facebook.com/tools/explorer/?method=GET
        &path=me&version=v20.0 and selecting the app you want to use to the get
        the page access token.
    long_lived_user_access_token
        Long-lived user access token.

    Returns
    -------
    Long-lived page access token.

    Notes
    -----
    See https://developers.facebook.com/docs/facebook-login/guides
    /access-tokens/get-long-lived/ for more details.
    """
    params = {"access_token": long_lived_user_access_token}
    req = HttpRequest(base_url=BASE_URL, endpoint=f"/{app_scoped_user_id}/accounts",
                      params=params)
    res = _send_request(req)
    return res.json()["data"][0]["access_token"]


def get_followers_count(username: str, instagram_account_id: int, access_token: str,
                        retry_initial_interval: int = 60 * 5,
                        retry_multiplier: float = 3,
                        max_tries: int = 5, retry_max_interval: int = 60 * 60):
    ENDPOINT = f"/{instagram_account_id}"
    PARAMS = {
        "fields": f"business_discovery.username({username}){{followers_count}}",
        "access_token": access_token,
    }
    req = HttpRequest(base_url=BASE_URL, endpoint=ENDPOINT, params=PARAMS)
    # only retry if 403 error, indicating too many requests
    # instagram returns 400 errors if access token is invalid
    res = retry(_send_request, initial_interval=retry_initial_interval,
                multiplier=retry_multiplier,
                max_tries=max_tries, max_interval=retry_max_interval,
                e=partial(is_http_error, status_code=(403,)),
                args=(req,)).json()
    return res["business_discovery"]["followers_count"]


def get_media(username: str, instagram_account_id: int, access_token: str,
              page_size: int = 200, start_time: Optional[datetime] = None,
              retry_initial_interval: int = 60 * 5,
              retry_multiplier: float = 3,
              max_tries: int = 5, retry_max_interval: int = 60 * 60):
    i = 0
    data = []
    after = None
    while True:
        logger.info(f"Getting media for {username}"
                    f" ({i * page_size} to {(i + 1) * page_size})")
        media_str = f"media.limit({page_size})"
        if after is not None:
            media_str += f".after({after})"
        params = {
            "fields": f"business_discovery.username({username})"
                      f"{{{media_str}{{comments_count,like_count,timestamp}}}}",
            "access_token": access_token,
        }
        req = HttpRequest(base_url=BASE_URL, endpoint=f"/{instagram_account_id}",
                          params=params)
        res = retry(_send_request, initial_interval=retry_initial_interval,
                    multiplier=retry_multiplier, max_tries=max_tries,
                    max_interval=retry_max_interval,
                    e=partial(is_http_error, status_code=(403,)),
                    args=(req,)).json()
        media = res["business_discovery"]["media"]
        page_data = media["data"]
        page_data = pd.DataFrame(page_data, columns=MEDIA_COLUMNS)
        page_data["timestamp"] = pd.to_datetime(
            page_data["timestamp"]).dt.tz_localize(None)
        page_data = page_data.set_index("timestamp").sort_index()
        if not page_data.empty:
            cursors = media["paging"]["cursors"]
            data.append(page_data.loc[start_time:])
            min_timestamp = page_data.index[0]
            if (start_time is not None and min_timestamp <= start_time) or (
                    "after" not in cursors):
                break
            after = cursors["after"]
        else:
            break
        i += 1
    return pd.concat(data).sort_index()
