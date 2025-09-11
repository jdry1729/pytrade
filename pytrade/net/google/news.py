import json
import logging
import re
from datetime import datetime, timedelta
from typing import Union, Tuple

import jsmin
import pandas as pd
from bs4 import BeautifulSoup
from pytrade.net.http import HttpRequest, _send_request
from pytrade.utils.retry import retry
from pytrade.utils.time import get_equally_spaced_times, sleep
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_URL = "https://news.google.com"
ENDPOINT = "/search"


def get_articles(query: str, start_time: datetime, end_time: datetime,
                 chunk_size: timedelta = timedelta(days=30),
                 chunk_delay: Union[int, Tuple[int, int]] = (0, 2),
                 show_progress: bool = True, retry_initial_interval: int = 60 * 5,
                 retry_max_interval: int = 60 * 60,
                 max_tries: int = 5, retry_multiplier: int = 3) -> pd.DataFrame:
    """
    Gets articles from Google News for a particular query.

    Parameters
    ----------
    query
        Query.
    start_time
        Start time to get articles from.
    end_time
        End time to get articles to.
    chunk_size
        Determines period each request gets data for. If set too large (e.g., to
        1 year), not all articles published over the period will be returned.
        Emprically, it seems that the maximum number of articles returned for a
        query is ~95. To capture as many articles as possible, you should use
        a small chunk size (e.g., 1W), but this will result in the function taking
        longer.
    chunk_delay
    show_progress
    retry_initial_interval
        Initial interval for retry.
    retry_max_interval
        Max interval for retry.
    max_tries
        Max attempts in event of error.
    retry_multiplier
        Retry multiplier.

    Returns
    -------
    Articles.
    """
    times = get_equally_spaced_times(start_time, end_time, period=chunk_size)
    if end_time not in times:
        times.append(end_time)

    data = []
    for i in tqdm(range(len(times) - 1), disable=not show_progress):
        if i > 0:
            sleep(chunk_delay)
        chunk_start_time = times[i]
        chunk_end_time = times[i + 1]
        # ensure period for which data is always downloaded equals chunk size
        # this ensures the "density" of the articles retrieved is constant
        chunk_start_time -= max([timedelta(),
                                 chunk_size + chunk_start_time -
                                 chunk_end_time])
        chunk_start_time_str = chunk_start_time.strftime('%Y-%m-%d')
        chunk_end_time_str = chunk_end_time.strftime('%Y-%m-%d')
        logger.info(f"Getting news articles for {query} between"
                    f" {chunk_start_time_str} and {chunk_end_time_str}")
        # mustn't set user agent since it triggers google consent redirect
        req = HttpRequest(
            base_url=BASE_URL,
            endpoint=ENDPOINT,
            params={
                "q": f"{query} after:{chunk_start_time_str}"
                     f" before:{chunk_end_time_str}",
            },
        )
        res = retry(_send_request, args=(req,), max_tries=max_tries,
                    initial_interval=retry_initial_interval,
                    max_interval=retry_max_interval,
                    multiplier=retry_multiplier)
        soup = BeautifulSoup(res.content, "html.parser")
        tag = soup.find(
            "script", string=lambda x: x and 'data:["gsrres"' in x)
        data_ = re.search(r"data:(\[.*\])", jsmin.jsmin(tag.text), re.DOTALL)
        articles = json.loads(data_.group(1))[1]
        if len(articles):
            for article in articles[0]:
                data.append({
                    "headline": article[0][2],
                    "time": article[0][4][0],
                    "link": article[0][6],
                })
    data = pd.DataFrame(data)
    data["time"] = pd.to_datetime(data["time"], unit="s")
    data = data.set_index("time").sort_index()
    # there may be duplicates in the data
    return data.loc[~data.duplicated(keep="first")]
