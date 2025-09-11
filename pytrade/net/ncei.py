import json
import re
from datetime import datetime

import jsmin
import pandas as pd
from pytrade.net.http import HttpRequest, _send_request

STATE_CODES = {
    "AL": "1",
    "AK": "50",
    "AZ": "2",
    "AR": "3",
    "CA": "4",
    "CO": "5",
    "CT": "6",
    "DE": "7",
    "DC": "49",
    "FL": "8",
    "GA": "9",
    "HI": "51",
    "ID": "10",
    "IL": "11",
    "IN": "12",
    "IA": "13",
    "KS": "14",
    "KY": "15",
    "LA": "16",
    "ME": "17",
    "MD": "18",
    "MA": "19",
    "MI": "20",
    "MN": "21",
    "MS": "22",
    "MO": "23",
    "MT": "24",
    "NE": "25",
    "NV": "26",
    "NH": "27",
    "NJ": "28",
    "NM": "29",
    "NY": "30",
    "NC": "31",
    "ND": "32",
    "OH": "33",
    "OK": "34",
    "OR": "35",
    "PA": "36",
    "PR": "66",
    "RI": "37",
    "SC": "38",
    "SD": "39",
    "TN": "40",
    "TX": "41",
    "UT": "42",
    "VT": "43",
    "VA": "44",
    "WA": "45",
    "WV": "46",
    "WI": "47",
    "WY": "48",
}


def get_statewide_metric(state: str, metric: str, start_time: datetime,
                         end_time) -> pd.Series:
    req = HttpRequest(
        base_url="https://www.ncei.noaa.gov",
        endpoint=f"/access/monitoring/climate-at-a-glance/statewide/time-series/"
                 f"{STATE_CODES[state]}/{metric}/1/0/"
                 f"{start_time.year}-{end_time.year}/zingchart-config.js",
    )
    res_ = _send_request(req)
    script_content = jsmin.jsmin(res_.text)
    match = re.search(
        r"var chartConfigchartCanvas\s*=\s*({.*?});", script_content, re.DOTALL
    )
    chart_config = json.loads(match.group(1))
    values = chart_config["series"][0]["values"]
    labels = chart_config["scaleX"]["labels"]
    data = pd.Series(dict(zip(labels, values)))
    data.index = pd.to_datetime(data.index)
    return data
