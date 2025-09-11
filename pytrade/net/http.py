from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Union, Collection, List, Any
from urllib.parse import urlparse, parse_qsl

import aiohttp
import requests
from curl_cffi import requests as requests_cffi
from requests import HTTPError
from retry.api import retry_call

logger = logging.getLogger(__name__)


class HttpMethod(enum.Enum):
    GET = 0
    POST = 1
    PUT = 2
    DELETE = 3


@dataclass
class URLParseResult:
    scheme: str
    hostname: str
    endpoint: str
    params: Dict[str, str] = None

    @property
    def base_url(self) -> str:
        return f"{self.scheme}://{self.hostname}"


def parse_url(url: str) -> URLParseResult:
    parse = urlparse(url)
    params = dict(parse_qsl(parse.query))
    return URLParseResult(scheme=parse.scheme, hostname=parse.hostname,
                          endpoint=parse.path, params=params)


@dataclass
class HttpRequest:
    base_url: str
    endpoint: str
    method: HttpMethod = HttpMethod.GET
    params: Optional[Dict[str, str]] = None
    data: Optional[Dict[str, str]] = None
    headers: Optional[Dict[str, str]] = None
    json: Optional[Union[Dict[str, Any], List]] = None
    cookies: Optional[Dict] = None
    auth: Optional = None
    verify: bool = True

    @property
    def url(self):
        return f"{self.base_url}{self.endpoint}"

    def to_dict(self) -> Dict:
        return {
            "method": self.method.name,
            "url": self.url,
            "params": self.params,
            "headers": self.headers,
            "data": self.data,
            "json": self.json,
            "cookies": self.cookies,
            "verify": self.verify
        }


async def aio_request(sess: aiohttp.ClientSession, request: HttpRequest):
    async with sess.request(str(request.method), request.url,
                            params=request.params, data=request.data,
                            headers=request.headers) as res:
        # TODO: return
        return await res.json()


def get_request_kwargs(request: HttpRequest):
    return {
        "method": request.method.name,
        "url": request.url,
        "params": request.params,
        "headers": request.headers,
        "data": request.data,
        "json": request.json,
        "cookies": request.cookies,
        "auth": request.auth,
        "verify": request.verify,
    }


def send_request_cffi(request: HttpRequest, raise_for_status: bool = True):
    res = requests_cffi.request(**get_request_kwargs(request))
    if raise_for_status:
        res.raise_for_status()
    return res


def _send_request(request: HttpRequest,
                  session: Optional[requests.Session] = None,
                  raise_for_status: bool = True):
    logger.debug(f"Sending request: {request}")
    kwargs = get_request_kwargs(request)
    if session is not None:
        res = session.request(**kwargs)
    else:
        res = requests.request(**kwargs)
    if raise_for_status:
        res.raise_for_status()
    return res


# TODO: set default for tries to -1?
# TODO: deprecate and use retry + _send_request above
def send_request(request: HttpRequest,
                 session: Optional[requests.Session] = None, tries: int = 1,
                 delay: int = 20, max_delay: int = 60, backoff=2):
    return retry_call(_send_request, [request, session],
                      tries=tries, delay=delay, max_delay=max_delay,
                      backoff=backoff, logger=logger)


def is_http_error(e: Exception,
                  status_code: Optional[Union[int, Collection[int]]] = None):
    if isinstance(e, HTTPError):
        if status_code is not None:
            if isinstance(status_code, int):
                return e.response.status_code == status_code
            return e.response.status_code in status_code
        return True
    return False


def get_public_ip() -> str:
    req = HttpRequest(base_url="https://httpbin.org",
                      endpoint="/ip")
    res = _send_request(req).json()
    return res["origin"]
