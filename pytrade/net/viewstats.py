import base64
import json
import struct

import pandas as pd
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from pytrade.net.constants import USER_AGENT
from pytrade.net.http import HttpMethod, HttpRequest, _send_request
from pytrade.utils.retry import retry

_IV = (
    "Wzk3LCAxMDksIC0xMDAsIC05MCwgMTIyLCAtMTI0LCAxMSwgLTY5LCAtNDIsIDExNSwgLTU4LCAtNjcs"
    "IDQzLCAtNzUsIDMxLCA3NF0="
)
_KEY = (
    "Wy0zLCAtMTEyLCAxNSwgLTEyNCwgLTcxLCAzMywgLTg0LCAxMDksIDU3LCAtMTI3LCAxMDcsIC00Niwg"
    "MTIyLCA0OCwgODIsIC0xMjYsIDQ3LCA3NiwgLTEyNywgNjUsIDc1LCAxMTMsIC0xMjEsIDg5LCAtNzEs"
    "IDUwLCAtODMsIDg2LCA5MiwgLTQ2LCA0OSwgNTZd"
)

BASE_URL = "https://api.viewstats.com"

COLUMN_DTYPES = {
    "id": str,
    "subscriberCount": int,
    "viewCount": int,
    "videoCount": int,
    "date": "datetime64[ns]",
    "longViews": float,
    "shortViews": float,
}

COLUMNS = ["insertedAt"] + list(COLUMN_DTYPES.keys())


def _decrypt_res(res: bytes) -> str:
    key = json.loads(base64.b64decode(_KEY))
    iv = json.loads(base64.b64decode(_IV))

    key = struct.pack('B' * len(key), *[(x + 256) % 256 for x in key])
    iv = struct.pack('B' * len(iv), *[(x + 256) % 256 for x in iv])

    cipher = Cipher(algorithms.AES(key), modes.GCM(iv, res[-16:]),
                    backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(res[:-16]) + decryptor.finalize()
    return decrypted.decode('utf-8')


def get_stats(handle: str, api_key: str, range_: str = "alltime",
              groupBy: str = "daily") -> pd.DataFrame:
    headers = {
        "User-Agent": USER_AGENT,
        "origin": "https://www.viewstats.com",
        "referer": "https://www.viewstats.com/",
        "authorization": f"Bearer {api_key}",
    }

    params = {
        "range": range_,
        "groupBy": groupBy,
        "sortOrder": "ASC",
        "withRevenue": "true",
        "withEvents": "true",
        "withBreakdown": "true",
        "withToday": "true",
    }

    req = HttpRequest(
        base_url=BASE_URL,
        endpoint=f"/channels/@{handle}/stats",
        method=HttpMethod.GET,
        params=params,
        headers=headers,
    )

    res = retry(_send_request, args=(req,))
    data = _decrypt_res(res.content)
    res = json.loads(data)

    data = pd.DataFrame(res["data"], columns=COLUMNS)
    data["date"] = pd.to_datetime(data["date"])
    data["insertedAt"] = pd.to_datetime(data["insertedAt"]).dt.tz_localize(None)
    data = data.set_index("insertedAt")

    data = data[COLUMN_DTYPES.keys()].astype(COLUMN_DTYPES)

    return data
