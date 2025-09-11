from datetime import datetime

import pandas as pd
from pytrade.net.constants import USER_AGENT
from pytrade.net.http import HttpRequest, HttpMethod, _send_request

BASE_URL = "https://corescientific.com"
ENDPOINT = "/wp-admin/admin-ajax.php"


def get_daily_btc_mined() -> pd.Series:
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint=ENDPOINT,
        method=HttpMethod.POST,
        data={"action": "btc_mined"},
        headers={"User-Agent": USER_AGENT}
    )
    res = _send_request(req).json()
    amount = float(res["amount"])
    time = datetime.strptime(res["date"], "%m/%d/%y")
    data = pd.Series({time: amount})
    data.index.name = "time"
    data.name = "btc_mined"
    return data
