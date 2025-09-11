import json
import logging
import re
from datetime import datetime
from typing import Optional

import jsmin
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from pytrade.net.constants import USER_AGENT
from pytrade.net.http import HttpRequest, _send_request
from pytrade.utils.pandas import empty_df, empty_time_idx
from pytrade.utils.retry import retry
from pytrade.utils.time import normalize
from tqdm import tqdm

nltk.download("punkt")

logger = logging.getLogger(__name__)

# set no_silent_downcasting to True so when doing fillna on column with all None
# values the dtype of the column won't change to float
pd.set_option('future.no_silent_downcasting', True)

EARNINGS_COLUMNS = [
    "date",
    "period_ending",
    "estimate",
    "reported",
    "surprise",
    "surprise_pct",
    "time"
]

ARTICLES_COLUMNS = {
    "title": str,
    "teaser": str,
    "body": str,
    "endpoint": str,
    "tags": str,
}

EVENT_TYPE_MAP = {
    "earnings": 1,
}

BASE_URL = "https://www.zacks.com"


def get_earning_surprise(symbol: str, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None):
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint=f"/stock/research/{symbol}/earnings-calendar",
        headers={
            "User-Agent": USER_AGENT,
        },
    )
    res = retry(_send_request, args=(req,))
    soup = BeautifulSoup(res.content, "html.parser")

    data = soup.find("script", string=lambda x: x and "document.obj_data" in x)
    script_content = jsmin.jsmin(data.text)

    match = re.search(r"document\.obj_data\s*=\s*({.*});", script_content,
                      re.DOTALL)
    data = json.loads(match.group(1))

    data = data["earnings_announcements_earnings_table"]
    data = pd.DataFrame(data, columns=EARNINGS_COLUMNS).drop(
        columns=["surprise", "surprise_pct"])
    data[["estimate", "reported"]] = data[["estimate", "reported"]].map(
        lambda x: x.replace('$', ''))

    data = data.replace("--", np.nan)
    data[["estimate", "reported"]] = data[["estimate", "reported"]].astype(float)
    data["surprise"] = data["reported"] - data["estimate"]

    data["date"] = pd.to_datetime(data["date"], format="%m/%d/%y")
    data["period_ending"] = pd.to_datetime(data["period_ending"], format="%m/%Y")
    data = data.set_index("date").sort_index()

    if start_time is not None:
        data = data.loc[start_time:]
    if end_time is not None:
        data = data.loc[:end_time]

    return data


def get_article_body(endpoint: str, retry_initial_interval: int = 60 * 5,
                     retry_max_interval: int = 60 * 60,
                     max_tries: int = 5, retry_multiplier: int = 3) -> str:
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint=endpoint,
        headers={
            "User-Agent": USER_AGENT,
        },
    )
    res = retry(_send_request, args=(req,), max_tries=max_tries,
                initial_interval=retry_initial_interval,
                max_interval=retry_max_interval, multiplier=retry_multiplier)
    soup = BeautifulSoup(res.content, "html.parser")
    return soup.find("div", {"class": "commentary_body"}).find("div", {
        "id": "comtext"}).decode_contents().strip()


def get_zacks_rating(body: str):
    # remove sentences containing below substrings
    ignore = [
        "stocks with",
        "stock-rating system",
        "positive earnings esp",
        "proprietary model",
    ]
    soup = BeautifulSoup(body)
    body = re.sub(r"\s\s+", " ", soup.get_text(separator=" ", strip=True))
    body = " ".join(
        x
        for x in sent_tokenize(body)
        if not any(y in x.lower() for y in ignore)
    )
    res = re.findall(r"Zacks Rank(?:\sof)? #\s?(\d+)", body)
    if len(res) == 1:
        return int(res[0])
    return np.nan


def get_articles(
        symbol: str, start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        retry_initial_interval: int = 60 * 5,
        retry_max_interval: int = 60 * 60,
        max_tries: int = 5, retry_multiplier: int = 3):
    data = []
    page = 1
    done = False
    num_pages = None
    while True:
        logger.info(f"Getting page {page} of articles for: {symbol}")
        req = HttpRequest(
            base_url=BASE_URL,
            endpoint=f"/stock/research/{symbol}/all-news",
            params={"page": page},
            headers={
                "User-Agent": USER_AGENT,
            },
        )
        res = retry(_send_request, args=(req,), max_tries=max_tries,
                    initial_interval=retry_initial_interval,
                    max_interval=retry_max_interval,
                    multiplier=retry_multiplier)
        soup = BeautifulSoup(res.content, "html.parser")

        content = soup.find("div", {"id": "content"})
        articles = content.find_all(
            "div", {"class": "listitem", "data-page-url": True})

        if not articles:
            logger.info(f"No articles found for: {symbol}")
            break

        if num_pages is None:
            num_pages = 1
            pagination = soup.find("div", {"id": "pagination"})
            if pagination is not None:
                num_pages = int(pagination.find_all("a")[-2].text)
            logger.info(f"Total pages: {num_pages}")

        # articles are ordered by time descending
        for article in articles:
            title = article.find("h3", {"class": "listing_article_heading"}).find(
                "a").text.strip()
            endpoint = article["data-page-url"]

            time = article.find("time").text.replace("Published on ", "")[:-1]
            time = datetime.strptime(time, "%B %d,%Y")
            if end_time is not None and time > end_time:
                logger.info(f"Skipping article since published after"
                            f" end time: {title}")
                continue
            if start_time is not None and time < start_time:
                done = True
                break

            teaser = article.find("p", {"class": "teaser"}).text
            tags = article.find_all("a", {"class": "article_tag"})
            tags = [x.text.strip() for x in tags]
            tags = ",".join(tags)

            # endpoint for some articles is invalid, so we need try-except below
            try:
                body = get_article_body(
                    endpoint, max_tries=max_tries,
                    retry_initial_interval=retry_initial_interval,
                    retry_max_interval=retry_max_interval,
                    retry_multiplier=retry_multiplier
                )
            except Exception:
                logger.warning(f"Error getting article: {title}")
                continue

            logger.info(f"Found article: {title} ({time.strftime('%Y-%m-%d')})")
            data.append(
                {
                    "time": time,
                    "title": title,
                    "teaser": teaser,
                    "body": body,
                    "endpoint": endpoint,
                    "tags": tags,
                }
            )
        page += 1
        if done or page > num_pages:
            break

    if data:
        data = pd.DataFrame(data)
        data = data.set_index("time").sort_index()
        data = data.astype(ARTICLES_COLUMNS)
    else:
        data = pd.DataFrame([], columns=list(ARTICLES_COLUMNS.keys()),
                            index=pd.DatetimeIndex([], name="time"))

    if start_time is not None:
        data = data.loc[start_time:]
    if end_time is not None:
        data = data.loc[:end_time]
    return data


# TODO: allow different types here to get guidance, etc.
def get_events(type_: str = "earnings",
               start_time: Optional[datetime] = None,
               end_time: Optional[datetime] = None) -> pd.DataFrame:
    if type_ != "earnings":
        raise ValueError("Error getting events; type_ must be \"earnings\"")

    if start_time is None:
        start_time = datetime.utcnow()
    if end_time is None:
        end_time = start_time
    start_time = normalize(start_time)
    end_time = normalize(end_time)

    res = []
    times = pd.date_range(start_time, end_time)
    for time in tqdm(times):
        req = HttpRequest(
            base_url="https://www.zacks.com",
            endpoint="/includes/classes/z2_class_calendarfunctions_data.php",
            params={
                "calltype": "eventscal",
                # must set hour to > 6 otherwise day before's results returned
                "date": int(time.replace(hour=8).timestamp()),
                "type": EVENT_TYPE_MAP[type_],
            },
            headers={
                "User-Agent": USER_AGENT
            }
        )
        res_ = _send_request(req)
        data_ = re.search(r"^window.app_data=(.*)", jsmin.jsmin(res_.text), re.DOTALL)
        data_ = json.loads(data_.group(1))

        for x in data_["data"]:
            soup = BeautifulSoup(x[0], "html.parser")
            symbol = soup.find('span', class_='hoverquote-symbol').get_text()
            release_time = x[3]
            estimate = x[4]
            res.append({"time": time, "symbol": symbol, "release_time": release_time,
                        "estimate": estimate, "reported": x[5]})

    if res:
        return pd.DataFrame(res).set_index("time")
    return empty_df(index=empty_time_idx(),
                    columns=["symbol", "release_time", "estimate", "reported"])
