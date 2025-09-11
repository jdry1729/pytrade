import logging
import sqlite3
from typing import List, Dict, Iterable, Callable, Tuple, Optional
from tqdm import tqdm

import secretstorage
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from selenium.webdriver.chrome.webdriver import WebDriver
from http.cookiejar import Cookie, CookieJar

from pytrade.utils.collections import get_first_index_greater_than_or_equal_to
from pytrade.utils.typing import DatetimeOrFloat
from urllib.parse import unquote

logger = logging.getLogger(__name__)

CHROME_SALT = b"saltysalt"
CHROME_V10_PASSWORD = "peanuts"


def _get_chrome_password(version) -> bytes:
    if version == "v10":
        return CHROME_V10_PASSWORD.encode("utf8")
    bus = secretstorage.dbus_init()
    collection = secretstorage.get_default_collection(bus)
    for item in collection.get_all_items():
        if item.get_label() == 'Chrome Safe Storage':
            return item.get_secret()
    raise ValueError("Error getting chrome password")


def decrypt_chrome_value(value: bytes) -> str:
    version = value[:3].decode("UTF-8")
    password = _get_chrome_password(version)
    key = PBKDF2(password, CHROME_SALT, 16, 1)
    cipher = AES.new(key, AES.MODE_CBC, IV=b" " * 16)
    res = cipher.decrypt(value[3:])
    # the last byte is repeated as many times as the integer value of the byte
    # it's just used as a filler, so must be removed before decoding the result
    if len(res):
        return res[:-res[-1]].decode("utf8")


def get_chrome_cookies(db: str, host_key: Optional[Tuple[str]] = None) -> List[Dict]:
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    params = ()
    stmt = "SELECT * FROM cookies"
    if host_key is not None:
        placeholders = ", ".join(["?"] * len(host_key))
        stmt += f" WHERE host_key IN ({placeholders})"
        params = host_key

    cursor.execute(stmt, params)
    column_names = [description[0] for description in cursor.description]

    rows = cursor.fetchall()
    rows = [dict(zip(column_names, row)) for row in rows]

    for x in tqdm(rows):
        x["value"] = decrypt_chrome_value(x.pop("encrypted_value"))
    return rows


def convert_chrome_cookies_to_webdriver_format(cookies) -> List[Dict]:
    cookies_ = []
    for x in cookies:
        cookie_ = {"name": x["name"], "value": x["value"], "path": x["path"],
                   "domain": x["host_key"], "httpOnly": bool(x["is_httponly"])}
        if x["has_expires"] == 1:
            cookie_["expiry"] = int((x["expires_utc"] / 1000000) - 11644473600)
        cookies_.append(cookie_)
    return cookies_


def _create_cookie(name, value, domain, path, secure, expires) -> Cookie:
    return Cookie(
        version=0,
        name=name,
        value=value,
        port=None,
        port_specified=False,
        domain=domain,
        domain_specified=True,
        domain_initial_dot=domain.startswith('.'),
        path=path,
        path_specified=True,
        secure=bool(secure),
        expires=expires,
        discard=False,
        comment=None,
        comment_url=None,
        rest={'HttpOnly': None},
    )


def convert_chrome_cookies_to_cookie_jar(cookies: List[Dict]):
    cookie_jar = CookieJar()

    cookies = [x for x in cookies if x["value"] is not None]
    for cookie in cookies:
        decoded_value = unquote(cookie["value"])
        new_cookie = _create_cookie(
            name=cookie["name"],
            value=decoded_value,
            domain=cookie["host_key"],
            path=cookie["path"],
            secure=cookie["is_secure"],
            expires=cookie["expires_utc"] // 10 ** 6
        )
        cookie_jar.set_cookie(new_cookie)
    return cookie_jar


def try_add_cookies(driver: WebDriver, cookies: Iterable[Dict]) -> None:
    # cookies can only be added if current url of driver matches domain of cookie
    for cookie in cookies:
        try:
            driver.add_cookie(cookie)
            logger.info(f"Added cookie: {cookie}")
        except:
            pass


def time_period_to_page_range(
        num_pages: int,
        span_fn: Callable[[int], Tuple[DatetimeOrFloat, DatetimeOrFloat]],
        start_time: DatetimeOrFloat, end_time: DatetimeOrFloat,
        ascending: bool = False) -> Tuple[int, int]:
    """
    Computes page range which spans period defined by start and end times.

    Parameters
    ----------
    num_pages
        Total number of pages.
    span_fn
        Function which accepts page number and returns tuple giving time period
        spanned by the page. Both times are inclusive. Could cache function return
        values if expensive to compute.
    start_time
        Start time, inclusive.
    end_time
        End time, exclusive.
    ascending
        Whether the data is ordered ascending or descending. If ascending, then
        page 1 contains the latest data, otherwise the last page does.

    Notes
    -----
    Assumes pages are ordered by time descending. So first page contains most
    recent data.

    Returns
    -------
    Page range tuple. First element is start page, second is end page. Both pages
    are inclusive. I.e., to get data for the entire time range, data for both
    start and end pages needs to be fetched. Start page might be 1, end page might
    be 10, for example.
    """
    # TODO: fix to work with ascending=True
    if ascending:
        raise ValueError("Error converting time period to range; ascending argument"
                         " must currently be false")
    logger.info(f"Computing page range for period: {start_time} to {end_time}")
    pages = list(range(num_pages, 0, -1))
    try:
        # if ascending=False, end page will be larger than start page
        end_idx = get_first_index_greater_than_or_equal_to(
            0, len(pages) - 1, start_time, value_fn=lambda idx: span_fn(pages[idx])[1])
        end_page = pages[end_idx]
        logger.info(f"Found end page: {end_page}")
    except ValueError:
        raise ValueError(
            "Error computing page range; start time after first page end time")
    try:
        start_idx = get_first_index_greater_than_or_equal_to(
            end_idx, len(pages) - 1, end_time,
            value_fn=lambda idx: span_fn(pages[idx])[1])
        start_page = pages[start_idx]
    except ValueError:
        start_page = pages[-1]
    if start_page == num_pages:
        if end_time <= span_fn(num_pages)[0]:
            raise ValueError(
                "Error computing page range; end time before last page start time")
    logger.info(f"Found start page: {start_page}")
    return start_page, end_page
