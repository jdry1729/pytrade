import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Collection

import pandas as pd
from pytrade.net.http import HttpMethod, HttpRequest, _send_request
from pytrade.net.keepa.selenium import _keepa_series_to_pandas
from pytrade.utils import stack
from pytrade.utils.retry import retry
from requests import HTTPError

BASE_URL = "https://api.keepa.com"

logger = logging.getLogger(__name__)


def keepa_to_datetime(keepa_time):
    if keepa_time is not None:
        return datetime.utcfromtimestamp((keepa_time + 21564000) * 60)


@dataclass
class Product:
    asin: str
    title: str
    brand: Optional[str]
    manufacturer: Optional[str]
    tracking_since: datetime
    product_group: Optional[str]
    release_date: Optional[datetime]
    variations: List[str]
    last_update: datetime
    root_category: int
    categories: List[int]
    price: pd.Series
    parent_rating: Optional[pd.Series] = None
    parent_rating_count: Optional[pd.Series] = None
    parent_asin: Optional[str] = None
    rating_count: Optional[pd.Series] = None
    review_count: Optional[pd.Series] = None
    sales_ranks: Optional[pd.Series] = None
    monthly_sold: Optional[pd.Series] = None


def _retry_format_fn(e: HTTPError) -> str:
    if e.response.status_code == 429:
        res = e.response.json()
        return f"429 error; tokens_left={res['tokensLeft']}"
    return str(e)


def get_asins(
        api_key: str,
        *,
        domain: int = 1,
        title: Optional[str] = None,
        brand: Optional[str] = None,
        manufacturer: Optional[str] = None,
        single_variation: bool = False,
        min_ratings: Optional[int] = None,
        page_size: int = 1000,
        pages: Optional[int] = None,
        sort_by: Optional[str] = None,
        ascending: bool = False,
        min_variation_count: Optional[int] = None,
        max_variation_count: Optional[int] = None,
        retry_initial_interval: int = 120,
        retry_max_interval: Optional[int] = None,
        max_tries: int = 10,
        retry_multiplier: int = 1,
) -> List[str]:
    """
    COUNT_REVIEWS_current
    """
    query = {"perPage": page_size, "singleVariation": single_variation}
    if title is not None:
        query["title"] = title
    if brand is not None:
        query["brand"] = brand
    if manufacturer is not None:
        query["manufacturer"] = manufacturer
    if min_ratings is not None:
        query["current_COUNT_REVIEWS_gte"] = min_ratings
    if sort_by is not None:
        query["sort"] = [[sort_by, "asc" if ascending else "desc"]]
    if min_variation_count is not None:
        query["variationCount_gte"] = min_variation_count
    if max_variation_count is not None:
        query["variationCount_lte"] = max_variation_count

    asins = []
    page = 0
    while True:
        logger.info(f"Getting page {page + 1} of results")
        query["page"] = page
        req = HttpRequest(
            base_url=BASE_URL,
            endpoint="/query",
            method=HttpMethod.GET,
            params={
                "key": api_key,
                "domain": domain,
                "selection": json.dumps(query),
            },
            headers={"Accept-Encoding": "gzip"},
        )

        res = retry(
            _send_request, args=(req,),
            initial_interval=retry_initial_interval,
            max_interval=retry_max_interval,
            max_tries=max_tries,
            multiplier=retry_multiplier,
            format_fn=_retry_format_fn
        ).json()
        if page == 0:
            # total results is total results matching query, not number of results
            # returned on page
            logger.info(f"Total results: {res['totalResults']}")

        asins_ = res["asinList"]
        asins.extend(asins_)
        if len(asins_) < page_size:
            break

        # can get max of 10000 results per query
        page += 1
        if (page + 1) * page_size > 10000:
            break
        if pages is not None and page >= pages:
            break

    return asins


def _get_products(
        api_key: str,
        asins: Collection[str],
        *,
        domain: int = 1,
        retry_initial_interval: int = 120,
        retry_max_interval: Optional[int] = None,
        max_tries: int = 10,
        retry_multiplier: int = 1,
        days: Optional[int] = None,
) -> List[Product]:
    params = {
        "key": api_key,
        "asin": ",".join(asins),
        "domain": domain,
        "history": 1,
        "rating": 1,
        "update": 12,
    }
    if days is not None:
        params["days"] = days
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint="/product",
        method=HttpMethod.GET,
        params=params,
        headers={"Accept-Encoding": "gzip"},
    )
    res = retry(
        _send_request,
        args=(req,),
        initial_interval=retry_initial_interval,
        max_interval=retry_max_interval,
        max_tries=max_tries,
        multiplier=retry_multiplier,
        format_fn=_retry_format_fn,
    ).json()

    products = []
    for product in res["products"]:

        parent_rating = None
        if product["csv"][16] is not None:
            parent_rating = _keepa_series_to_pandas(product["csv"][16])

        parent_rating_count = None
        if product["csv"][17] is not None:
            parent_rating_count = _keepa_series_to_pandas(product["csv"][17])

        rating_count = None
        review_count = None
        if "reviews" in product:
            reviews = product["reviews"]
            if reviews.get("ratingCount") is not None:
                rating_count = _keepa_series_to_pandas(reviews["ratingCount"])
            if reviews.get("reviewCount") is not None:
                review_count = _keepa_series_to_pandas(reviews["reviewCount"])

        sales_rank = None
        if product.get("salesRanks") is not None:
            sales_rank = stack(
                {
                    k: _keepa_series_to_pandas(v)
                    for k, v in product["salesRanks"].items()
                },
                names=["category"],
            )

        monthly_sold = None
        if product.get("monthlySoldHistory") is not None:
            monthly_sold = _keepa_series_to_pandas(
                product["monthlySoldHistory"]
            )

        variations = []
        if product.get("variationCSV") is not None:
            variations = product["variationCSV"].split(",")

        products.append(
            Product(
                price=_keepa_series_to_pandas(product["csv"][0]),
                tracking_since=keepa_to_datetime(product["trackingSince"]),
                brand=product["brand"],
                title=product["title"],
                manufacturer=product["manufacturer"],
                asin=product["asin"],
                product_group=product["productGroup"],
                release_date=keepa_to_datetime(product["releaseDate"]),
                parent_asin=product["parentAsin"],
                variations=variations,
                last_update=keepa_to_datetime(product["lastUpdate"]),
                root_category=product["rootCategory"],
                categories=product["categories"],
                parent_rating=parent_rating,
                parent_rating_count=parent_rating_count,
                rating_count=rating_count,
                review_count=review_count,
                sales_ranks=sales_rank,
                monthly_sold=monthly_sold,
            )
        )

    return products


def get_products(
        api_key: str,
        asins: Collection[str],
        *,
        domain: int = 1,
        retry_initial_interval: int = 120,
        retry_max_interval: Optional[int] = None,
        max_tries: int = 10,
        retry_multiplier: int = 1,
        days: Optional[int] = None
) -> List[Product]:
    products = []
    for i in range(0, len(asins), 100):
        logger.info(f"Getting data for ASINs {i + 1} to {i + 100}")
        products.extend(_get_products(
            api_key, asins[i: i + 100], domain=domain,
            retry_initial_interval=retry_initial_interval,
            retry_max_interval=retry_max_interval,
            max_tries=max_tries,
            retry_multiplier=retry_multiplier, days=days)
        )
    return products
