import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union, Iterable

import pandas as pd
from pytrade.net.constants import USER_AGENT
from pytrade.net.http import HttpMethod, HttpRequest, _send_request, is_http_error
from pytrade.utils.collections import ensure_list
from pytrade.utils.functions import partial
from pytrade.utils.retry import retry

logger = logging.getLogger(__name__)

BASE_URL_1 = "https://www.reddit.com"
BASE_URL_2 = 'https://oauth.reddit.com'

REDDIT_LINK_KIND = "t3"


@dataclass
class Credentials:
    username: str
    password: str
    client_id: str
    client_secret: str


POSTS_COLUMN_TYPES = {
    "id": str,
    "subreddit": str,
    "title": str,
    "author": str,
    "selftext": "object",
    "downs": int,
    "ups": int,
    "link_flair_text": "object",
    "score": int,
    "num_comments": int,
    "url": str,
}

POST_COLUMNS_EXT = list(POSTS_COLUMN_TYPES) + ["created_utc"]


# TODO: cache for an hour?
def get_auth_token(credentials: Credentials) -> str:
    data = {
        "grant_type": "password",
        "username": credentials.username,
        "password": credentials.password,
    }
    req = HttpRequest(
        base_url=BASE_URL_1,
        endpoint="/api/v1/access_token",
        method=HttpMethod.POST,
        data=data,
        auth=(credentials.client_id, credentials.client_secret),
        headers={
            "User-Agent": USER_AGENT,
        }
    )
    res = _send_request(req).json()
    logger.info(f"Created {res['token_type']} access token; expires in"
                f" {res['expires_in']}s")
    return res["access_token"]


def _get_posts(query: str, subreddit: str, auth_token: str, *,
               start_time: Optional[datetime] = None,
               retry_initial_interval: int = 60 * 5,
               retry_multiplier: float = 3,
               max_tries: int = 5, retry_max_interval: int = 60 * 60) -> pd.DataFrame:
    posts = []
    after = None
    params = {
        "q": query,
        "restrict_sr": 1,
        "sort": "new",
        "limit": 100,
    }
    i = 0
    while True:
        logger.info(f"Getting page {i + 1} of search results; {query=}, {subreddit=}")
        if after is not None:
            params["after"] = after
        req = HttpRequest(
            base_url=BASE_URL_2,
            # TODO: need .json below?
            endpoint=f"/r/{subreddit}/search.json",
            headers={"Authorization": auth_token, "User-Agent": USER_AGENT},
            params=params
        )
        res = retry(_send_request,
                    initial_interval=retry_initial_interval,
                    multiplier=retry_multiplier,
                    max_tries=max_tries, max_interval=retry_max_interval,
                    e=partial(is_http_error, status_code=(429,)),
                    args=(req,)).json()
        posts_ = [{col: x["data"][col] for col in POST_COLUMNS_EXT} for x in
                  res["data"]["children"] if x.get("kind") == REDDIT_LINK_KIND]
        posts_ = pd.DataFrame(posts_, columns=POST_COLUMNS_EXT)
        posts_["created_utc"] = pd.to_datetime(posts_["created_utc"], unit="s")
        posts_ = posts_.set_index("created_utc").sort_index()
        posts.append(posts_.astype(POSTS_COLUMN_TYPES))
        if res["data"]["after"] is None or (
                start_time is not None and min(posts_.index) <= start_time):
            break
        after = res["data"]["after"]
        i += 1
    if posts:
        return pd.concat(posts).sort_index().loc[start_time:]
    return pd.DataFrame([], columns=list(POSTS_COLUMN_TYPES),
                        index=pd.DatetimeIndex([], name="created_utc"))


def get_posts(query: str, subreddit: Union[str, Iterable[str]], auth_token: str,
              *, start_time: Optional[datetime] = None, exact: bool = False,
              retry_initial_interval: int = 60 * 5,
              retry_multiplier: float = 3,
              max_tries: int = 5, retry_max_interval: int = 60 * 60) -> pd.DataFrame:
    posts = []
    subreddits_ = []
    subreddits = ensure_list(subreddit)
    for subreddit in subreddits:
        posts_ = _get_posts(query, subreddit, auth_token,
                            start_time=start_time,
                            retry_initial_interval=retry_initial_interval,
                            retry_multiplier=retry_multiplier,
                            max_tries=max_tries,
                            retry_max_interval=retry_max_interval,
                            )
        if exact:
            posts_ = posts_.loc[posts_["title"].str.contains(query) |
                                posts_["selftext"].str.contains(query)]
        if not posts_.empty:
            posts.append(posts_)
            subreddits_.append(subreddit)
    if posts:
        return pd.concat(posts).sort_index()
    return pd.DataFrame([], columns=list(POSTS_COLUMN_TYPES),
                        index=pd.DatetimeIndex([], name="created_utc"))

# def get_post_comments(post_id: str, *, credentials: Credentials):
#     comments = []
#     after = None
#     auth_token = get_auth_token(credentials)
#     params = {
#         "restrict_sr": 1,
#         "sort": "new",
#         "limit": 100,
#     }
#     i = 0
#     while True:
#         logger.info(f"Getting page {i + 1} of search results; {post_id=}")
#         if after is not None:
#             params["after"] = after
#         req = HttpRequest(
#             base_url=BASE_URL_2,
#             endpoint=f"/comments/{post_id}",
#             headers={"Authorization": auth_token, "User-Agent": USER_AGENT},
#             params=params
#         )
#         res = retry(_send_request,
#                     cond_fn=partial(is_http_error, status_code=(429,)),
#                     args=(req,)).json()
#         # TODO: finish
#     return res
