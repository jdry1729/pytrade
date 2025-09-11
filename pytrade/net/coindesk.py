import json
import logging
import math
import time
from datetime import date
from typing import Optional, Tuple, Union

import pandas as pd
from curl_cffi import requests
from pytrade.net.http import HttpMethod, HttpRequest
from pytrade.utils.retry import retry
from pytrade.utils.time import sleep

BASE_URL = "https://www.coindesk.com"

logger = logging.getLogger(__name__)

COLUMNS = [
    "_id",
    "title",
    "description",
    "link",
    "pubdateunix",
    "creator",
    "subheadlines",
    "subtype",
    "type",
    "section",
    "sponsored"
]


def get_articles(session: requests.Session, query: str, *,
                 start_date: Optional[date] = None,
                 end_date: Optional[date] = None, section: Optional[str] = None,
                 page_delay: Optional[Union[int, Tuple[int, int]]] = None,
                 retry_initial_interval: int = 1,
                 retry_max_interval: int = 30,
                 max_tries: int = 1, retry_multiplier: int = 1):
    """
    Gets all articles for a query between start and end dates inclusive.
    """
    search_query = query
    query = {"search_query": search_query, "sort": 1}

    filter_url = ""
    if start_date is not None:
        start_time_str = start_date.strftime("%-m/%-d/%Y")
        end_time_str = end_date.strftime("%-m/%-d/%Y")
        filter_url = f"&daterange={start_time_str},{end_time_str}"
    if section is not None:
        filter_url += f"&facetedkey=section|&facetedvalue={section}|"

    if filter_url:
        query["filter_url"] = filter_url

    def send_request(request: HttpRequest):
        res = session.request(**request.to_dict())
        try:
            res.raise_for_status()
        except Exception as e:
            raise Exception(res.content) from e
        return res.json()

    def get_page(page):
        query["page"] = page
        req = HttpRequest(
            base_url=BASE_URL,
            method=HttpMethod.GET,
            endpoint="/pf/api/v3/content/fetch/search",
            params={"query": json.dumps(query)},
        )

        return retry(send_request, initial_interval=retry_initial_interval,
                     max_interval=retry_max_interval, max_tries=max_tries,
                     multiplier=retry_multiplier, request=req)

    total_results = get_page(0)["metadata"]["total"]
    total_pages = math.ceil(total_results / 10)

    articles = []
    for page in range(total_pages):
        logger.info(f"Getting data for: {search_query} (page {page + 1}"
                    f" of {total_pages})")
        data = get_page(page)
        for item in data["items"]:
            articles.append({k: v for k, v in item.items() if k in COLUMNS})
        if page_delay is not None:
            sleep(page_delay)

    articles = pd.DataFrame(articles, columns=COLUMNS)
    articles["pubdateunix"] = pd.to_datetime(articles["pubdateunix"], unit="s")
    return articles.set_index("pubdateunix").sort_index()
