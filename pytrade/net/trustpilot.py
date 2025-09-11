import logging
import re
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
from bs4 import BeautifulSoup

from pytrade.net.constants import USER_AGENT
from pytrade.net.http import _send_request, HttpRequest, is_http_error
from pytrade.net.utils import time_period_to_page_range
from pytrade.utils.constants import MIN_TIME
from pytrade.utils.functions import partial
from pytrade.utils.retry import retry
from pytrade.utils.time import sleep

logger = logging.getLogger(__name__)

BASE_URL = "https://uk.trustpilot.com"

REVIEWS_DTYPES = {
    "id": "object",
    "rating": "int",
    "text": "object",
}


def _get_page(page: int, business_unit: str) -> pd.DataFrame:
    data = []
    logger.info(f"Getting reviews for {business_unit}; {page=}")
    endpoint = f"/review/{business_unit}?sort=recency"
    req = HttpRequest(base_url=BASE_URL, endpoint=endpoint,
                      params={"page": str(page)},
                      headers={"User-Agent": USER_AGENT})
    res = retry(_send_request, args=(req,), initial_interval=300,
                multiplier=2, max_interval=600,
                e=partial(is_http_error, status_code=(403, 429, 500, 502)))
    soup = BeautifulSoup(res.content, "html.parser")
    reviews = soup.find_all("section", class_=re.compile(
        "^styles_reviewContentwrapper.*"))
    for review in reviews:
        header = review.find("div",
                             class_=re.compile("^styles_reviewHeader.*"))
        rating = header["data-service-review-rating"]
        time = pd.Timestamp(header.find("time")["datetime"]).tz_localize(None)
        content = review.find("div",
                              class_=re.compile("^styles_reviewContent.*"))
        text = content.find("p").text
        title_tag = content.find("a")
        # review likely marked harmful if no a tag exists, so isn't displayed
        if title_tag is not None:
            review_id = title_tag["href"]
            review_id = review_id.replace("/reviews/", "")
            data.append({
                "id": review_id,
                "rating": int(rating),
                "time": time,
                "text": text,
            })
    return pd.DataFrame(
        data, columns=["id", "rating", "time", "text"]).set_index("time").sort_index()


# TODO: cache?
def _page_span_fn(page: int, business_unit: str) -> Tuple[datetime, datetime]:
    data = _get_page(page, business_unit)
    if not data.empty:
        return data.index.min(), data.index.max()
    raise ValueError("Error computing page span; no data")


def get_reviews(business_unit: str, *, start_page: Optional[int] = None,
                end_page: Optional[int] = None,
                start_time: Optional[datetime] = None,
                end_time: Optional[datetime] = None,
                delay: int = 2) -> pd.DataFrame:
    """
    Gets Trustpilot reviews by parsing the DOM.

    Parameters
    ----------
    business_unit
        Business unit to get reviews for, e.g., "www.tesco.com".
    start_page
        Start page to get reviews from.
    end_page
        End page to get reviews to.
    delay
        Delay between making requests.
    start_time
        Start time to get data for.
    end_time
        End time to get data to.

    Returns
    -------
    Dataframe of reviews.

    Notes
    -----
    Trustpilot banned my IP after making ~200 requests in 5 minutes.
    """
    is_page_range = True
    if start_time is not None or end_time is not None:
        is_page_range = False
        if start_page is not None or end_page is not None:
            raise ValueError("Error getting reviews; neither start_page nor end_page"
                             " can be specified if either start_time or end_time is")

    data = []
    endpoint = f"/review/{business_unit}?sort=recency"
    req = HttpRequest(base_url=BASE_URL, endpoint=endpoint,
                      params={"page": "1"})
    # trustpilot returns 403 error if too many requests
    res = retry(_send_request, args=(req,), initial_interval=300,
                multiplier=2, max_interval=600,
                e=partial(is_http_error, status_code=(403, 429, 500, 502)))
    soup = BeautifulSoup(res.content, "html.parser")
    pagination_buttons = soup.find_all("a", {
        "name": re.compile("pagination-button-([1-9][0-9]*|last)")})
    total_pages = max(int(x.text) for x in pagination_buttons)
    reviews = soup.find_all("section", class_=re.compile(
        "^styles_reviewContentwrapper.*"))

    if reviews:

        if is_page_range:
            if start_page is None:
                start_page = 1
            if end_page is None:
                end_page = total_pages
            end_page = min(end_page, total_pages)
        else:
            if start_time is None:
                start_time = MIN_TIME
            if end_time is None:
                end_time = datetime.utcnow()
            try:
                start_page, end_page = time_period_to_page_range(
                    total_pages, partial(_page_span_fn, business_unit=business_unit),
                    start_time=start_time, end_time=end_time)
            except ValueError:
                # either start time after first page end time or end time before last
                # page start time
                logger.warning("No reviews found in time range")

        if start_page is not None and end_page is not None:
            logger.info(f"Getting data from page {start_page} to page {end_page}")
            for page in range(start_page, end_page + 1):
                data.append(_get_page(page, business_unit))
                if page < end_page:
                    sleep(delay)

            data = pd.concat(data).sort_index()
            if start_time is not None:
                data = data.loc[start_time:]
            if end_time is not None:
                data = data.loc[:end_time]

            return data

    data = pd.DataFrame([], columns=list(REVIEWS_DTYPES.keys()),
                        index=pd.DatetimeIndex([], name="time"))
    return data.astype(REVIEWS_DTYPES)
