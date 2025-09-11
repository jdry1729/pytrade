import json
import logging
import re
import time
from datetime import datetime
from typing import Optional, Dict

import pandas as pd
import urllib3
from bs4 import BeautifulSoup
from jsmin import jsmin
from selenium.webdriver.chrome.webdriver import WebDriver

logger = logging.getLogger(__name__)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "https://seekingalpha.com"

ARTICLES_DTYPES = {"id": "string",
                   "commentCount": int,
                   "title": "string",
                   "authorId": "string",
                   "primaryTickers": "string",
                   "secondaryTickers": "string",
                   "link": "string"}


def get_article(driver: WebDriver, article_id: str) -> Dict:
    logger.info(f"Getting article: {article_id}")
    driver.get(f"https://seekingalpha.com/article/{article_id}")

    soup = BeautifulSoup(driver.page_source, features="lxml")
    script = soup.find("script", string=lambda x: x and "window.SSR_DATA" in x)
    script = jsmin(script.text)
    data_ = json.loads(
        re.search(r"window\.SSR_DATA\s*=\s*({.*});", script, re.DOTALL).group(1)
    )["article"]["response"]

    relationships = data_["data"]["relationships"]
    attributes = data_["data"]["attributes"]

    included = data_["included"]
    tags = ",".join(x["attributes"]["name"] for x in included if x["type"] == "tag")
    sentiments = [x["attributes"]["type"] for x in included if x["type"] == "sentiment"]

    if len(sentiments) > 1:
        raise ValueError("Error getting article; mutliple sentiments found")

    publish_on = attributes["publishOn"]
    publish_on = pd.Timestamp(publish_on).tz_convert("UTC").tz_localize(None)
    last_modified = attributes["lastModified"]
    last_modified = pd.Timestamp(last_modified).tz_convert("UTC").tz_localize(None)

    return {
        "id": article_id,
        "publishOn": publish_on,
        "authorId": relationships["author"]["data"]["id"],
        "title": attributes["title"],
        "summary": " ".join(attributes["summary"]),
        "content": attributes["content"],
        "lastModified": last_modified,
        "tags": tags,
        "sentiments": ",".join(sentiments),
        "commentCount": attributes["commentCount"],
        "likesCount": attributes["likesCount"],
    }


def get_articles(driver: WebDriver, ticker: str,
                 start_time: Optional[datetime] = None, delay: int = 10):
    """
    Gets Seeking Alpha articles for a ticker using Selenium.

    Parameters
    ----------
    driver
        Driver to use. You must use Chrome in undetected mode. The easiest way
        to do this is to use Selenium Base.
    ticker
        Ticker.
    start_time
        Start time.
    delay
        Delay.

    Returns
    -------
    Articles.

    Examples
    --------
    ```
    from seleniumbase import Driver
    driver = Driver(uc=True, headless=False)
    get_articles_slnm(driver, "VFC", datetime(2024, 1, 1))
    ```
    """
    page = 1

    articles = []
    while True:
        logger.info(f"Getting articles for {ticker}; {page=}")
        driver.get(f"https://seekingalpha.com/symbol/{ticker}/analysis?page={page}")

        soup = BeautifulSoup(driver.page_source, features="lxml")
        script = soup.find("script", string=lambda x: x and "window.SSR_DATA" in x)
        script = jsmin(script.text)
        data_ = json.loads(
            re.search(r"window\.SSR_DATA\s*=\s*({.*});", script, re.DOTALL).group(1)
        )["analysis"]["response"]["data"]

        if data_:
            articles_ = []
            for x in data_:
                attrs = x["attributes"]
                rels = x["relationships"]
                articles_.append({
                    "id": x["id"],
                    "publishOn": attrs["publishOn"],
                    "commentCount": attrs["commentCount"],
                    "title": attrs["title"],
                    "authorId": rels["author"]["data"]["id"],
                    "primaryTickers": ",".join(
                        x["id"] for x in rels["primaryTickers"]["data"]),
                    "secondaryTickers": ",".join(
                        x["id"] for x in rels["secondaryTickers"]["data"]),
                    "link": x["links"]["self"],
                })
            articles_ = pd.DataFrame(articles_)
            articles_["publishOn"] = articles_["publishOn"].apply(
                lambda x: pd.Timestamp(x).tz_convert("UTC").tz_localize(None))
            articles_ = articles_.set_index("publishOn").sort_index()

            articles.append(articles_)

            if start_time is not None and articles_.index[0] < start_time:
                break
        else:
            break

        logger.info(f"Sleeping for {delay}s before getting next page")
        time.sleep(delay)
        page += 1

    articles = pd.concat(articles).sort_index().loc[start_time:]
    return articles.reindex(columns=list(ARTICLES_DTYPES.keys())).astype(
        ARTICLES_DTYPES)
