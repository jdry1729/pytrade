import logging
from datetime import datetime

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_URL = "http://s3-us-west-1.amazonaws.com/umbrella-static"


def get_rankings_for_date(date: datetime):
    date_str = date.strftime("%Y-%m-%d")
    logger.info(f"Getting rankings for date: {date_str}")
    data = pd.read_csv(f"{BASE_URL}/top-1m-{date_str}.csv.zip", header=None)
    data.columns = ["rank", "url"]
    data["time"] = date
    return data.set_index(["time", "rank"])["url"]


def get_rankings(start_time: datetime, end_time: datetime):
    logger.info(f"Getting rankings from {start_time} to {end_time}")
    rankings = []
    times = pd.date_range(start_time, end_time)
    for time in tqdm(times):
        rankings.append(get_rankings_for_date(time))
    return pd.concat(rankings)
