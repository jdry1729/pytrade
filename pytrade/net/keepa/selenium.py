import base64
import json
import logging
import math
import urllib.parse
import zlib
from typing import Optional, Union, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from pytrade.net.webdriver import get_performance_logs, element_exists
from pytrade.utils.pandas import empty_df, empty_idx
from pytrade.utils.pandas import stack
from pytrade.utils.time import sleep
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

logger = logging.getLogger(__name__)

PRODUCT_DTYPES = {
    "title": "string",
    "brand": "string",
    "manufacturer": "string",
    "product_group": "string",
    "release_date": "datetime64[ns]",
    "price": float,
    "rating": "Int64",
    "review_count": "Int64",
    "tracking_since": "datetime64[ns]",
    "parent_asin": "string",
    "variations": "string",
    "last_update": "datetime64[ns]",
    "root_category": "Int64",
    "categories": "string",
}

QUOTA_EXCEEDED_XPATH = "//div[@id='popupTitle' and contains(text(), 'Quota exceeded')]"


def _keepa_series_to_pandas(s: List, dtype: Any = float) -> pd.Series:
    data = pd.DataFrame(np.reshape(s, (-1, 2)), columns=["time", "value"])
    data = data.astype({"time": int, "value": dtype})
    data["time"] = pd.to_datetime((data["time"] + 21564e3) * 60, unit="s")
    return data.set_index("time")["value"].sort_index()


def _decode_message(message, wbits=-zlib.MAX_WBITS):
    compressed_data = base64.b64decode(message)
    decompressed_data = zlib.decompress(compressed_data, wbits=wbits)
    return decompressed_data.decode("utf-8")


def get_token_status(driver: WebDriver, timeout: int = 30) -> Dict:
    driver.get("https://keepa.com")
    sleep(1)
    get_performance_logs(driver)
    driver.get("https://keepa.com/#!finder")
    sleep(0.1)
    WebDriverWait(driver, timeout).until(
        EC.invisibility_of_element_located((By.XPATH, "//div[@role='progressbar']"))
    )
    sleep(1)
    logs = get_performance_logs(driver)
    messages = _get_ws_messages(logs)
    requests = [x for x in messages if x.get("type") == "getTokenStatus"]
    if len(requests) != 1:
        raise ValueError("Error getting token status; unknown request ID")
    request_id = requests[0]["id"]

    responses = [x for x in messages if x["id"] == request_id and "tokens" in x]
    if len(responses) != 1:
        raise ValueError("Error getting token status; unknown response")
    response = responses[0]
    return {"tokens": response["tokens"],
            "token_bucket_size": response["tokenBucketSize"]}


def wait_for_tokens(driver: WebDriver, tokens: int, timeout: int = 30) -> None:
    status = get_token_status(driver, timeout)
    logger.info(f"Number of tokens: {status['tokens']}")
    if status["tokens"] >= tokens:
        return
    delay = math.ceil((tokens - status["tokens"]) / 1000.0)
    logger.info(f"Sleeping for {delay}h so tokens greater than {tokens}")
    sleep(delay * 60 ** 2)


def login(driver, username: str, password: str) -> None:
    login_xpath = "//span[@id='panelUserRegisterLogin']"
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, login_xpath))
    )
    driver.find_element(By.XPATH, login_xpath).click()
    sleep(3)

    email_xpath = "//input[@name='username']"
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, email_xpath))
    )
    driver.find_element(By.XPATH, email_xpath).send_keys(username)
    password_xpath = "//input[@name='password']"
    sleep(3)

    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, password_xpath))
    )
    driver.find_element(By.XPATH, password_xpath).send_keys(password)

    driver.find_element(By.XPATH, "//input[@id='submitLogin']").click()

    sleep(10)


def _click_next_page(driver):
    driver.find_element(
        By.XPATH, "//div[@class='ag-paging-button' and @ref='btNext']").click()


def get_finder_url(
        *,
        brand: Optional[str] = None,
        min_ratings: Optional[int] = None,
        manufacturer: Optional[str] = None,
        sort_col: Optional[str] = None,
        single_variation_per_product: bool = True,
        ascending: bool = False,
):
    data = {"f": {}}
    if brand is not None:
        data["f"]["brand"] = {
            "filterType": "autocomplete",
            "filter": brand,
            "type": "isOneOf",
        }
    if min_ratings is not None:
        data["f"]["COUNT_REVIEWS_current"] = {
            "filterType": "number",
            "type": "greaterThanOrEqual",
            "filter": min_ratings, "filterTo": None}
    if manufacturer is not None:
        data["f"]["manufacturer"] = {
            "filterType": "autocomplete",
            "filter": manufacturer,
            "type": "isOneOf",
        }
    if single_variation_per_product:
        data["f"]["singleVariation"] = {
            "filterType": "boolean",
            "filter": "on",
            "type": "equals",
        }
    if sort_col is not None:
        data["s"] = [
            {"colId": sort_col, "sort": "asc" if ascending else "desc"}
        ]
    data["t"] = "g"
    url = "https://keepa.com/#!finder/" + urllib.parse.quote(
        json.dumps(data, separators=(",", ":")), safe=":/!#"
    ).replace(":", "%3A")
    return url


def change_page_size(driver, page_size: int) -> None:
    wait_for_tokens(driver, 2000)
    driver.get("https://keepa.com/#!")
    url = get_finder_url(brand="amazon basics", min_ratings=100000,
                         single_variation_per_product=True)
    driver.get(url)
    sleep(3)
    tool_xpath = "//span[contains(@class, 'tool__row')]"
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, tool_xpath))
    ).click()
    item_xpath = f"//div[@id='tool-row-menu']//li[@data-value={page_size}]"
    WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, item_xpath))
    )
    driver.find_element(By.XPATH, item_xpath).click()
    logger.info(f"Changed page size to: {page_size}")


def _get_ws_messages(logs) -> List[Dict]:
    messages = []
    for log in logs:
        if log["method"] == "Network.webSocketFrameSent":
            message = _decode_message(
                log["params"]["response"]["payloadData"], zlib.MAX_WBITS
            )
        elif log["method"] == "Network.webSocketFrameReceived":
            message = _decode_message(
                log["params"]["response"]["payloadData"], -zlib.MAX_WBITS
            )
        else:
            continue
        messages.append(json.loads(message))
    return messages


def _wait_for_progress_bar(driver, timeout: Optional[int] = 30):
    sleep(0.1)
    WebDriverWait(driver, timeout).until(
        EC.invisibility_of_element_located((By.XPATH, "//div[@role='progressbar']"))
    )
    sleep(1)


def get_products(
        driver,
        *,
        pages: Optional[int] = None,
        brand: Optional[str] = None,
        min_ratings: Optional[int] = None,
        manufacturer: Optional[str] = None,
        sort_col: Optional[str] = None,
        single_variation_per_product: bool = True,
        ascending: bool = False,
        delay: Union[int, Tuple[int, int]] = (10, 20),
        timeout: int = 30
):
    """
    sort_col may be: COUNT_REVIEWS_current, SALES_current.

    Notes
    -----
    If single variation per product is True, Keepa seems to return whichever variation
    has the longest tracking history.
    """
    get_performance_logs(driver)

    # product results won't be displayed if already on finder page when you go to
    # finder url, so must go to home page first
    driver.get("https://keepa.com/#!")

    url = get_finder_url(
        brand=brand,
        min_ratings=min_ratings,
        manufacturer=manufacturer,
        sort_col=sort_col,
        single_variation_per_product=single_variation_per_product,
        ascending=ascending
    )
    driver.get(url)

    page = 1
    products = []
    while True:
        logger.info(
            f"Getting page {page} of results for query: {brand=}, {manufacturer=}")
        _wait_for_progress_bar(driver, timeout)
        if element_exists(driver, (By.XPATH, QUOTA_EXCEEDED_XPATH), timeout=timeout):
            raise ValueError("Error getting products; quota exceeded")

        logs = get_performance_logs(driver)
        messages = _get_ws_messages(logs)

        i = 0
        request_ids = [x["id"] for x in messages if x.get("path") == "pro/product"]
        for message in messages:
            if message["id"] in request_ids and "products" in message:
                product_ = message["products"][0]
                product = {
                    "price": product_["stats"]["current"][0],
                    "rating": product_["stats"]["current"][16],
                    "review_count": product_["stats"]["current"][17],
                    "tracking_since": product_["trackingSince"],
                    "brand": product_["brand"],
                    "title": product_["title"],
                    "manufacturer": product_["manufacturer"],
                    "asin": product_["asin"],
                    "product_group": product_["productGroup"],
                    "release_date": product_["releaseDate"],
                    "parent_asin": product_["parentAsin"],
                    "variations": product_["variationCSV"],
                    "last_update": product_["lastUpdate"],
                    "root_category": product_["rootCategory"],
                    "categories": ",".join(str(x) for x in product_["categories"]),
                }
                products.append(product)
                i += 1
        logger.info(f"Found {i} products on page {page}")
        if pages is not None and page >= pages:
            break
        else:
            page += 1
            next_loc = (By.XPATH, "//div[@class='ag-paging-button' and @ref='btNext']")
            if element_exists(driver, next_loc, timeout=1):
                sleep(delay)
                driver.find_element(*next_loc).click()
            else:
                logger.info("No more pages found")
                break

    if products:
        products = pd.DataFrame(products).set_index("asin")
        products[["tracking_since", "last_update"]] = (
                products[["tracking_since", "last_update"]] * 60 + 21564e3 * 60
        ).apply(pd.to_datetime, unit="s", errors="coerce")
        # need coerce below since many products don't have release date
        products["release_date"] = pd.to_datetime(
            products["release_date"], format="%Y%m%d", errors="coerce")
        return products.reindex(columns=list(PRODUCT_DTYPES.keys())).astype(
            PRODUCT_DTYPES)
    return empty_df(index=empty_idx(name="asin"),
                    columns=list(PRODUCT_DTYPES.keys())).astype(
        PRODUCT_DTYPES)


def get_product(driver: WebDriver, asin: str, domain: str = "1",
                timeout: int = 30) -> Dict:
    get_performance_logs(driver)
    sleep(3)

    driver.get(f"https://keepa.com/#!product/{domain}-{asin}")
    _wait_for_progress_bar(driver, timeout)
    sleep(3)

    logs = get_performance_logs(driver)
    messages = _get_ws_messages(logs)

    # TODO: sometimes 2 requests are made for different ASINs; should explore why
    #  this is rather than just ignoring request made for other ASIN
    request_ids = [x["id"] for x in messages if x.get("path") == "product" and
                   x.get("asin") == asin]
    if len(request_ids) != 1:
        raise ValueError("Error getting product details; unknown websocket"
                         " request ID")

    request_id = request_ids[0]
    messages = [x for x in messages if x["id"] == request_id and "products" in x]
    if len(messages) != 1:
        raise ValueError("Error getting product details; unknown websocket response")
    return messages[0]["products"][0]


def get_product_metrics(driver: WebDriver, asin: str, domain: str = "1",
                        timeout: int = 30) -> Dict[
    str, pd.Series]:
    product = get_product(driver, asin, domain, timeout=timeout)

    metrics = {
        "parent_rating": _keepa_series_to_pandas(product["csv"][16]),
        "parent_rating_count": _keepa_series_to_pandas(product["csv"][17]),
    }

    if "reviews" in product:
        reviews = product["reviews"]
        metrics["rating_count"] = _keepa_series_to_pandas(reviews["ratingCount"])
        metrics["review_count"] = _keepa_series_to_pandas(reviews["reviewCount"])

    if "salesRanks" in product:
        metrics["sales_ranks"] = stack(
            {k: _keepa_series_to_pandas(v) for k, v in product["salesRanks"].items()},
            names=["category"],
        )

    if "monthlySoldHistory" in product:
        metrics["monthly_sold"] = _keepa_series_to_pandas(
            product["monthlySoldHistory"])

    return metrics
