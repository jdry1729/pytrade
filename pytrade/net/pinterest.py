import json
import logging
import urllib.parse
from datetime import datetime, timedelta
from typing import Union, Collection, Optional, List

import pandas as pd
from pytrade.net.webdriver import get_performance_logs, get_fetch_xhr_responses
from pytrade.utils.pandas import empty_df, empty_time_idx
from pytrade.utils.retry import retry
from pytrade.utils.time import sleep, normalize
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from tqdm import tqdm

logger = logging.getLogger(__name__)

TRENDS_DTYPES = {
    "count": float,
    "normalizedCount": float,
    "predictedUpperBoundNormalizedCount": float,
    "predictedLowerBoundNormalizedCount": float
}

BASE_URL = "https://trends.pinterest.com"


def login(driver: WebDriver, username: str, password: str) -> None:
    driver.get("http://www.pinterest.com")

    login_xpath = "//button[.//*[text()='Log in']]"
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, login_xpath))
    )
    driver.find_element(By.XPATH, login_xpath).click()

    email_xpath = "//input[@type='email']"
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, email_xpath))
    )
    driver.find_element(By.XPATH, email_xpath).send_keys(username)
    password_xpath = "//input[@type='password']"

    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, password_xpath))
    )
    driver.find_element(By.XPATH, password_xpath).send_keys(password)

    driver.find_element(
        By.XPATH,
        "//div[@data-test-id='fullPageSignupModal']//button[.//*[text()='Log in']]",
    ).click()

    sleep(10)
    logger.info("Successfully logged in")


def get_trends(
        driver: WebDriver,
        terms: Union[str, Collection[str]],
        country: str,
        *,
        end_time: Optional[datetime] = None,
        days: int = 730,
        normalize_against_group: bool = False,
) -> pd.DataFrame:
    """
    Gets trends data.

    Parameters
    ----------
    driver
    terms
    country
    end_time
        End time to use when fetching results. Must be at least 1 day ago. Although
        at certain times of the week, results only seem to be returned if set to
        2 or 3 days ago.
    days
        Number of days to get data for.
    normalize_against_group

    Returns
    -------
    Trends data.
    """
    if end_time is None:
        end_time = normalize(datetime.utcnow() - timedelta(days=1))

    normalize_against_group = "true" if normalize_against_group else "false"
    params = {
        "terms": ",".join(terms),
        "aggregation": 2,
        "days": days,
        "country": country,
        "end_date": end_time.strftime("%Y-%m-%d"),
        "normalize_against_group": normalize_against_group,
    }

    # empty performance logs
    get_performance_logs(driver)

    driver.get(f"{BASE_URL}/metrics/?{urllib.parse.urlencode(params)}")
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//pre"))
    )
    data = json.loads(driver.find_element(By.XPATH, "//pre").text)

    logs = get_performance_logs(driver)
    res = get_fetch_xhr_responses(logs, base_url=BASE_URL, endpoint="/metrics/")
    if len(res) > 1:
        raise ValueError("Error getting trends data; multiple responses received")

    # pinterest return empty json array even if 429/ 400 error sent - so we must
    # check response status code is 200
    res = res[0]
    status = res["params"]["response"]["status"]
    if status != 200:
        raise ValueError(f"Error getting trends data; response has {status} code")

    data_ = []
    for e in data:
        data_.extend([{**x, "term": e["term"]} for x in e["counts"]])

    data = pd.DataFrame(data_)
    if not data.empty:
        data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
        data = data.set_index(["date", "term"]).sort_index()
        data = data["count"].unstack().reindex(columns=terms)
        return data.astype(float)
    return empty_df(index=empty_time_idx("date"),
                    columns=pd.Index(terms, name="term")).astype(float)


def get_trends_1d(driver: WebDriver, terms: Union[str, Collection[str]],
                  country: str, *, end_time: Optional[datetime] = None,
                  days: int = 730, normalize_against_group: bool = True,
                  max_retries: int = 3, retry_initial_interval: int = 60,
                  retry_max_interval: int = 3 * 60,
                  retry_multiplier: float = 2) -> pd.DataFrame:
    data = []
    if end_time is None:
        end_time = normalize(datetime.utcnow() - timedelta(days=1))
    for i in tqdm(range(7)):
        trends = retry(
            get_trends, max_tries=max_retries,
            initial_interval=retry_initial_interval,
            max_interval=retry_max_interval, multiplier=retry_multiplier,
            args=(driver, terms), country=country,
            end_time=end_time - timedelta(days=i), days=days,
            normalize_against_group=normalize_against_group,
        )
        data.append(trends)
    data = pd.concat(data).sort_index()
    if data.index.duplicated().any():
        raise ValueError("Trends data contains duplicates")
    return data


def get_terms(driver: WebDriver, query: str, country: str) -> List[str]:
    params = {
        "query": query,
        "country": country
    }
    driver.get(
        f"https://trends.pinterest.com/prefix_match/?{urllib.parse.urlencode(params)}"
    )
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//pre"))
    )
    data = json.loads(driver.find_element(By.XPATH, "//pre").text)
    return [x["term"] for x in data]
