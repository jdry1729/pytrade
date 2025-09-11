import enum
import json
import logging
import os
import time
import urllib.parse
from datetime import timedelta, datetime
from typing import Tuple, Union, Optional, Iterable

import numpy as np
import pandas as pd
import requests
from pytrade.net.http import send_request, HttpRequest, HttpMethod
from pytrade.net.webdriver import element_exists
from pytrade.utils.files import wait_for_file
from pytrade.utils.retry import retry
from pytrade.utils.time import sleep, get_equally_spaced_times
from selenium.common import TimeoutException
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_URL = "https://trends.google.com/trends"

CHUNK_SIZE = {
    # approx 20 years
    "1M": pd.Timedelta("7300D"),
    # approx 5 years
    "1W": pd.Timedelta("1800D"),
    "1D": pd.Timedelta("200D"),
    "1H": pd.Timedelta("6D"),
    "8min": pd.Timedelta("30h"),
}

MIN_CHUNK_SIZE = {
    # approx 5 years
    "1M": pd.Timedelta("1828D"),
    "1W": pd.Timedelta("365D"),
    "1D": pd.Timedelta("7D"),
    "1H": pd.Timedelta("3D"),
    "8min": pd.Timedelta("1h"),
}

INDEX_NAME_MAP = {
    "1M": "Month",
    "1W": "Week",
    "1D": "Day",
    "1H": "Time",
    "8min": "Time",
}

# timedelta can't be created for 1M
TIMEDELTA_MAP = {
    "1M": "31D",
}

INDEX_FREQ_MAP = {
    "1M": "MS",
}

# must have space after widget-container below
ERROR_TITLE_XPATH = ("//div[contains(@class, 'widget-container ')][1]"
                     "//p[contains(@class, 'widget-error-title')]")

NOT_ENOUGH_DATA_TITLE = "Hmm, your search doesn't have enough data to show here."


class ErrorType(enum.Enum):
    SOMETHING_WENT_WRONG = 0
    NOT_ENOUGH_DATA = 1


class GoogleTrendsException(Exception):
    def __init__(self, message: str, error_type: ErrorType):
        super().__init__(message)
        self.message = message
        self.error_type = error_type


def accept_cookies(driver):
    """
    Accepts cookies if they haven't been already.
    """
    driver.get(f"{BASE_URL}/explore")
    try:
        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
            (By.CSS_SELECTOR,
             ".cookieBarButton.cookieBarConsentButton"))).click()
    except TimeoutException:
        logger.info("Cookies already accepted")


def compute_scaling_factor(search_vol: pd.DataFrame,
                           prev_search_vol: pd.DataFrame,
                           raise_if_no_overlapping_rows: bool) -> float:
    # TODO: use column with larger numbers if multiple columns
    #  have same number of non-nan values
    scaling_factor = prev_search_vol / search_vol
    scaling_factor = scaling_factor.replace([0, np.inf, -np.inf], np.nan)
    # only use single keyword for estimating scaling factor
    scaling_factor = scaling_factor[
        (~scaling_factor.isnull()).sum().idxmax()].dropna()
    num_overlapping_rows = len(scaling_factor)
    if num_overlapping_rows == 0:
        if raise_if_no_overlapping_rows:
            raise ValueError("Error computing scaling factor; no overlapping rows")
        else:
            logger.warning("Not re-scaling search volume since no"
                           " overlapping rows")
            return 1
    logger.info(f"Computing scaling factor based on {num_overlapping_rows}"
                f" rows present in current and previous search volume data")
    return scaling_factor.mean()


def _get_search_volume(session: requests.Session, keywords, start_time,
                       end_time):
    # TODO: if time delta is > 7 days, you can't specify minutes and seconds
    time_diff = end_time - start_time
    dt_format = "%Y-%m-%d"
    if time_diff < timedelta(days=7):
        dt_format += "T%H\\:%M\\:%S"
    start_time_str = start_time.strftime(dt_format)
    end_time_str = end_time.strftime(dt_format)
    req = {"comparisonItem": [],
           "category": 0,
           "property": "",
           }
    for keyword in keywords:
        req["comparisonItem"].append({
            "keyword": keyword,
            "time": f"{start_time_str} {end_time_str}",
            "geo": "",
        })
    params_1 = {'hl': 'en-GB', 'tz': 0, 'req': json.dumps(req)}
    req_1 = HttpRequest(method=HttpMethod.POST, base_url=BASE_URL,
                        endpoint="/api/explore", params=params_1)
    res_1 = send_request(req_1, session, tries=20, delay=3, max_delay=60,
                         backoff=2)
    widget = [x for x in json.loads(res_1.text[5:])["widgets"] if
              x["id"] == "TIMESERIES"][0]

    params_2 = {"req": json.dumps(widget["request"]), "token": widget["token"],
                "tz": 0}
    req_2 = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                        endpoint="/api/widgetdata/multiline", params=params_2)
    res_2 = send_request(req_2, session,
                         tries=20, delay=3, max_delay=60, backoff=2)
    data = json.loads(res_2.text[6:])["default"]["timelineData"]
    search_volume = pd.DataFrame(
        [{"time": int(x["time"]), **dict(zip(keywords, x["value"]))} for x in
         data], columns=["time", *keywords])
    search_volume["time"] = pd.to_datetime(search_volume['time'], unit='s')
    return search_volume.set_index("time")


def get_search_volume(keywords, start_time, end_time, freq="1D", *,
                      scale_periods: int = 20,
                      chunk_delay: Union[int, Tuple[int, int]] = (0, 30), ):
    results = []
    chunk_size = CHUNK_SIZE[freq]
    times = pd.date_range(start_time, end_time,
                          freq=chunk_size).to_pydatetime().tolist()
    sess = requests.Session()
    init_req = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                           endpoint="/explore")
    # first get cookies
    send_request(init_req, sess, tries=3, delay=1)
    if start_time not in times:
        times = [start_time] + times
    if end_time not in times:
        times += [end_time]
    prev_search_vol = None
    for i in tqdm(range(len(times) - 1)):
        if i > 0:
            sleep(chunk_delay)
        chunk_start_time = times[i] - pd.Timedelta(freq) * scale_periods
        chunk_end_time = times[i + 1]
        logger.info(f"Getting data between: {chunk_start_time} and"
                    f" {chunk_end_time}")
        search_vol = _get_search_volume(sess, keywords, chunk_start_time,
                                        chunk_end_time)
        num_zeroes = (search_vol.values == 0).sum()
        logger.info(f"Number of zero entries in search volume: {num_zeroes}")
        if prev_search_vol is not None and not prev_search_vol.empty:
            scaling_factor = compute_scaling_factor(search_vol,
                                                    prev_search_vol,
                                                    raise_if_no_overlapping_rows=True)
            logger.info(f"Re-scaling search volume using factor:"
                        f" {scaling_factor:.2f}")
            search_vol *= scaling_factor
        results.append(search_vol)
        prev_search_vol = search_vol
    data = pd.concat(results)
    # remove rows in scaling period
    data = data[~data.index.duplicated(keep="last")]
    return data.resample(freq).last().loc[start_time:end_time]


def _get_search_volume_slnm(driver, keywords, start_time, end_time,
                            download_dir: str, *, max_file_wait: int = 10,
                            max_page_load_wait: int = 10, geo: Optional[str] = None):
    file_path = os.path.join(download_dir, "multiTimeline.csv")
    logger.info(f"Temporarily downloading search data to: {file_path}")
    if os.path.exists(file_path):
        os.remove(file_path)

    time_diff = end_time - start_time
    dt_format = "%Y-%m-%d"
    if time_diff < timedelta(days=7):
        dt_format += "T%H\\:%M\\:%S"
        time_col = "Time"
    elif time_diff < timedelta(days=365):
        time_col = "Day"
    elif time_diff < timedelta(days=365 * 5):
        time_col = "Week"
    else:
        time_col = "Month"

    start_time_str = start_time.strftime(dt_format)
    end_time_str = end_time.strftime(dt_format)

    params = {"date": f"{start_time_str} {end_time_str}",
              "q": ",".join(keywords)}
    if geo is not None:
        params["geo"] = geo
    url = f"{BASE_URL}/explore?{urllib.parse.urlencode(params)}"
    logger.info(f"Getting data from: {url}")

    params = {"behavior": "allow", "downloadPath": download_dir}
    driver.execute_cdp_cmd("Page.setDownloadBehavior", params)
    driver.get(url)
    if element_exists(driver, (By.XPATH, ERROR_TITLE_XPATH), timeout=5):
        error_title = driver.find_element(By.XPATH, ERROR_TITLE_XPATH).text
        error_type = (ErrorType.NOT_ENOUGH_DATA if error_title == NOT_ENOUGH_DATA_TITLE
                      else ErrorType.SOMETHING_WENT_WRONG)
        raise GoogleTrendsException(f"Error getting search data;"
                                    f" error={error_type.name}", error_type)

    WebDriverWait(driver, max_page_load_wait).until(EC.element_to_be_clickable(
        (By.CSS_SELECTOR, ".widget-actions-item.export"))).click()
    wait_for_file(file_path, timeout=timedelta(seconds=max_file_wait))

    search_vol = pd.read_csv(file_path, skiprows=2)
    search_vol[time_col] = pd.to_datetime(search_vol[time_col])
    search_vol = search_vol.set_index(time_col)
    search_vol = search_vol.replace("<1", "0")
    search_vol = search_vol.astype(float)
    # remove ": (Worldwide)" or ": (United States" from column names
    search_vol.columns = search_vol.columns.str.replace(r":\s*\(.*\)$", "", regex=True)
    os.remove(file_path)
    return search_vol


def _take_screenshot(driver: WebDriver, screnshot_dir: str):
    timestamp = int(time.time())
    screenshot_path = os.path.join(screnshot_dir, f"{timestamp}.png")
    driver.save_screenshot(screenshot_path)
    logger.info(f"Saved screenshot to: {screenshot_path}")


def _get_chunk(driver, keywords, start_time: datetime, end_time: datetime, freq,
               driver_download_dir, driver_screenshot_dir, chunk_max_file_wait,
               chunk_max_page_load_wait, chunk_max_zero_prop,
               geo: Optional[str] = None) -> pd.DataFrame:
    try:
        search_vol = _get_search_volume_slnm(
            driver, keywords, start_time, end_time,
            download_dir=driver_download_dir,
            max_file_wait=chunk_max_file_wait,
            max_page_load_wait=chunk_max_page_load_wait,
            geo=geo
        )
        # TODO: check search_vol.index.inferred_freq is expected freq?
    except Exception as e:
        if driver_screenshot_dir is not None:
            _take_screenshot(driver, driver_screenshot_dir)
        if (isinstance(e, GoogleTrendsException)
                and e.error_type == ErrorType.NOT_ENOUGH_DATA):
            logger.warning("Not enough search data for period; returning all"
                           " zero data")
            # must specify 0.0 below as float
            return pd.DataFrame(
                0.0, columns=list(keywords),
                index=pd.DatetimeIndex(
                    pd.date_range(start_time, end_time,
                                  freq=INDEX_FREQ_MAP.get(freq, freq)),
                    name=INDEX_NAME_MAP[freq]))
        raise ValueError("Error getting search volume data") from e
    else:
        zero_count = (search_vol == 0).sum()
        keyword_zero_prop = zero_count / len(search_vol)
        # log zero proportions since google trends sometimes returns data with
        # lots of zeros in if you'd made too many requests
        logger.info(f"Keyword zero proportions:\n{keyword_zero_prop}")
        chunk_zero_prop = zero_count.sum() / search_vol.size
        logger.info(f"Chunk zero proportion: {chunk_zero_prop:.2f}")
        if driver_screenshot_dir is not None:
            _take_screenshot(driver, driver_screenshot_dir)
        if not search_vol.empty:
            if chunk_max_zero_prop is None or (
                    chunk_zero_prop < chunk_max_zero_prop):
                return search_vol
            raise ValueError("Error getting search volume data; max zero-proportion"
                             " exceeded")
    raise ValueError("Error getting search volume data")


def login(driver: WebDriver, email: str, password: str) -> None:
    driver.get(f"{BASE_URL}/explore")
    driver.refresh()

    sign_in_xpath = "//div[@id='one-google']//a[starts-with(@aria-label, 'Sign in')]"
    driver.find_element(By.XPATH, sign_in_xpath).click()

    email_xpath = "//input[@type='email']"
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
        (By.XPATH, email_xpath)))
    driver.find_element(By.XPATH, email_xpath).send_keys(email)

    driver.find_element(By.XPATH, "//div[@id='identifierNext']").click()

    password_xpath = "//input[@type='password']"
    WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
        (By.XPATH, password_xpath)))
    driver.find_element(By.XPATH, password_xpath).send_keys(password)

    driver.find_element(By.XPATH, "//div[@id='passwordNext']").click()

    account_xpath = ("//div[@id='one-google']//a[starts-with(@aria-label,"
                     " 'Google Account:')]")
    if not element_exists(driver, (By.XPATH, account_xpath)):
        raise ValueError("Error logging in")

    logger.info("Successsfully logged in")


def get_search_volume_slnm(
        driver: WebDriver,
        keywords: Iterable[str],
        start_time: datetime,
        end_time: datetime,
        freq="1D",
        *,
        driver_download_dir: Optional[str] = None,
        driver_screenshot_dir: Optional[str] = None,
        chunk_scale_periods: int = 20,
        chunk_delay: Union[int, Tuple[int, int]] = (0, 30),
        chunk_max_file_wait: int = 10,
        chunk_max_page_load_wait: int = 10,
        chunk_max_retries: int = 5,
        chunk_max_zero_prop: Optional[float] = None,
        raise_if_no_overlapping_rows: bool = True,
        geo: Optional[str] = None,
):
    """
    Gets search volume from Google Trends using selenium. Data is downloaded
    in chunks. The size of each chunk is 100 days if freq="1H", 6 days if
    freq="1H" and 30 hours if freq="8min".

    Parameters
    ----------
    driver
        Driver to use.
    keywords
        Keywords to download data for.
    start_time
        Start time to download data from (inclusive).
    end_time
        End time to download data to (inclusive).
    freq
        Frequency of data to download.
    driver_download_dir
        Where to download files to.
    driver_screenshot_dir
        Location to store screenshots of pages scraped.
    chunk_scale_periods
        Periods to use when stitching data together.
    chunk_delay
        Time to sleep for after making each request.
    chunk_max_file_wait
        Max time to wait for each file to download.
    chunk_max_page_load_wait
        Max time to wait for the export button to load.
    chunk_max_retries
        Maximum attempts to download each chunk of data.
    chunk_max_zero_prop
        Chunk maximum zero proportion.
    raise_if_no_overlapping_rows
        Raises an exception if no overlapping rows.
    geo
        What geography to get data for.

    Notes
    -----
    First request to google trends sometimes gives 429 error, so you should make
    this request prior to calling this function, and also accept cookies using the
    accept_cookies function.

    If this function fails to download the desired data, it's worth retrying
    with a driver that uses a different Chrome user data dir. Or, perhaps more easily,
    deleting the specified user data dir before re-trying.

    It seems drivers run not in headless mode give more accurate results.

    The value for time 2024-03-09 12:00, for example, represents the search volume for
    the period 2024-03-09 12:00 to 2024-03-09 13:00. Likewise, for 1D data, the value
    for time 2024-03-09 represents the search volume for the period 2024-03-09 to
    2024-03-10.

    Returns
    -------
    Search volume.
    """
    if driver_download_dir is None:
        driver_download_dir = os.path.expanduser("~/selenium")
    if not os.path.exists(driver_download_dir):
        os.makedirs(driver_download_dir)

    # TODO: if you set chunk scaling periods too high, you will get lower freq
    #  data than you asked for, should fix
    results = []
    times = get_equally_spaced_times(start_time, end_time, period=CHUNK_SIZE[freq])
    if end_time not in times:
        times.append(end_time)

    prev_search_vol = None
    for i in tqdm(range(len(times) - 1)):
        if i > 0:
            sleep(chunk_delay)

        chunk_start_time = times[i]
        chunk_end_time = times[i + 1]
        if i > 0:
            # for chunks after the first, we subtract freq * chunk_scale_periods
            # from the start time to ensure there is an overlap with previous chunk
            chunk_start_time -= pd.Timedelta(
                TIMEDELTA_MAP.get(freq, freq)) * chunk_scale_periods
        # make sure chunk size is sufficiently large so we don't download
        # higher frequency data than desired
        chunk_start_time -= max([timedelta(),
                                 MIN_CHUNK_SIZE[freq] + chunk_start_time -
                                 chunk_end_time])

        search_vol = retry(
            _get_chunk,
            initial_interval=30,
            max_tries=chunk_max_retries,
            multiplier=2,
            max_interval=60,
            driver=driver,
            keywords=keywords,
            start_time=chunk_start_time,
            end_time=chunk_end_time,
            freq=freq,
            driver_download_dir=driver_download_dir,
            driver_screenshot_dir=driver_screenshot_dir,
            chunk_max_file_wait=chunk_max_file_wait,
            chunk_max_page_load_wait=chunk_max_page_load_wait,
            chunk_max_zero_prop=chunk_max_zero_prop,
            geo=geo,
        )

        if prev_search_vol is not None:
            scaling_factor = compute_scaling_factor(
                search_vol, prev_search_vol,
                raise_if_no_overlapping_rows=raise_if_no_overlapping_rows)
            logger.info(f"Re-scaling search volume using factor:"
                        f" {scaling_factor:.2f}")
            search_vol *= scaling_factor

        results.append(search_vol)
        prev_search_vol = search_vol

    data = pd.concat(results)
    data = data[~data.index.duplicated(keep="last")]
    return data.loc[start_time:end_time]
