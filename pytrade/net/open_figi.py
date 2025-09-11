import logging
from typing import Optional, Dict, List

import requests
import retry

from pytrade.utils.profile import load_profile

BASE_URL = "https://api.openfigi.com"

logger = logging.getLogger(__name__)


# openfigi impose a max number of requests each minute so, if any error,
# we wait for 1 minute
@retry.retry(tries=3, delay=60)
def _make_request(endpoint: str, data: Dict) -> requests.models.Response:
    profile = load_profile()
    headers = {"X-OPENFIGI-APIKEY": profile.openfigi_api_key}
    res = requests.post(f"{BASE_URL}{endpoint}", json=data,
                        headers=headers)
    res.raise_for_status()
    return res


def search_figis(query: Optional[str] = None, exch_code: Optional[str] = None,
                 mic_code: Optional[str] = None,
                 security_type: Optional[str] = None):
    results = []
    data = {}
    if query is not None:
        data["query"] = query
    if exch_code is not None:
        data["exchCode"] = exch_code
    if mic_code is not None:
        data["micCode"] = mic_code
    if security_type is not None:
        data["securityType"] = security_type
    while True:
        res = _make_request("/v3/filter", data).json()
        results.extend(res["data"])
        if "next" not in res:
            break
        data["start"] = res["next"]
    return results


def map_to_figi(to_map: List[Dict]):
    """
    Examples
    --------
    map_to_figi([{"idType": "ID_ISIN", "idValue": "US0378331005"}])
    """
    # TODO: return dataframe?
    res = _make_request("/v3/mapping", to_map).json()
    return res
