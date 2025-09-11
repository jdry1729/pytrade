from datetime import datetime, time
from typing import Tuple, List, Optional, Collection

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from pytrade.utils.constants import MIN_TIME

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]


def get_fiscal_quarters(start: Tuple[int, str],
                        end: Tuple[int, str]) -> List[Tuple[int, str]]:
    res = []

    year = start[0]
    quarter = int(start[1][1])
    end_year = end[0]
    end_quarter = int(end[1][1])

    while year < end_year or (year == end_year and quarter <= end_quarter):
        res.append((year, f"Q{quarter}"))
        quarter += 1
        if quarter > 4:
            quarter = 1
            year += 1

    return res


def shift_fiscal_period(
        fiscal_period: Tuple[int, str], periods: int) -> Tuple[int, str]:
    year, quarter = fiscal_period
    current_quarter_index = QUARTERS.index(quarter)
    new_quarter_index = (current_quarter_index + periods) % 4
    new_year = year + (current_quarter_index + periods) // 4
    return new_year, QUARTERS[new_quarter_index]


def _get_local_trade_schedule(
        calendar: str, start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        trade_times: Optional[Collection[time]] = None) -> pd.DatetimeIndex:
    """
    Gets local trade schedule.

    Parameters
    ----------
    calendar
        Calendar.
    trade_times
        Trade times. Assumed to be in calendar's timezone.
    start_time
        Start time. Shouldn't have timezone specified, but assumed to be in calendar's
        timezone.
    end_time
        End time. Shouldn't have timezone specified, but assumed to be in calendar's
        timezone.

    Returns
    -------
    Local trade schedule.
    """
    calendar = mcal.get_calendar(calendar)

    if start_time is None:
        start_time = pd.Timestamp(MIN_TIME)
    if end_time is None:
        end_time = pd.Timestamp.now(tz=calendar.tz).tz_localize(None)

    dates = calendar.valid_days(start_date=start_time.date(),
                                end_date=end_time.date(), tz=None)
    if trade_times is None:
        schedule = dates
    else:
        schedule = pd.DatetimeIndex([datetime.combine(d, t) for d in dates
                                     for t in sorted(trade_times)])
    schedule = schedule[(start_time <= schedule) & (schedule <= end_time)]
    return schedule.rename("time")


def _get_utc_trade_schedule(
        calendar: str, start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        trade_times: Optional[
            Collection[time]] = None) -> pd.DatetimeIndex:
    """
    Gets UTC trade schedule.

    Parameters
    ----------
    calendar
        Calendar.
    trade_times
        Trade times. Shouldn't have timezone specified, but assumed to be in calendar's
        timezone.
    start_time
        Start time. Shouldn't have timezone specified, but assumed to be in UTC.
    end_time
        End time. Shouldn't have timezone specified, but assumed to be in UTC.

    Returns
    -------
    UTC trade schedule.
    """
    if start_time is None:
        start_time = pd.Timestamp(MIN_TIME)
    if end_time is None:
        end_time = pd.Timestamp.now(tz="UTC").tz_localize(None)

    calendar_ = mcal.get_calendar(calendar)
    start_time = pd.Timestamp(start_time).tz_localize("UTC")
    end_time = pd.Timestamp(end_time).tz_localize("UTC")
    local_start_time = start_time.tz_convert(calendar_.tz).tz_localize(None)
    local_end_time = end_time.tz_convert(calendar_.tz).tz_localize(None)
    local_schedule = _get_local_trade_schedule(
        calendar, local_start_time, local_end_time, trade_times)
    return local_schedule.tz_localize(
        calendar_.tz).tz_convert("UTC").tz_localize(None)


def get_trade_schedule(calendar: str, start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       trade_times: Optional[Collection[time]] = None,
                       utc: bool = False) -> pd.DatetimeIndex:
    if utc:
        return _get_utc_trade_schedule(calendar, start_time, end_time, trade_times)
    return _get_local_trade_schedule(calendar, start_time, end_time, trade_times)


def quarterly_yoy_pct_change(s: pd.Series, periods: int = 1) -> pd.Series:
    df = s.to_frame("value")
    df["quarter"] = df.index.to_period("Q")

    res = {}
    for q in df["quarter"].unique():
        q_curr = df[df["quarter"] == q]["value"]
        q_prev = df[df["quarter"] == (q - 4 * periods)]["value"]

        if len(q_prev) == 0 or len(q_curr) == 0:
            res[q] = np.nan
            continue

        q_prev.index = q_prev.index + pd.DateOffset(years=periods)

        aligned = pd.concat([
            q_curr.rename("curr"),
            q_prev.rename("prev")
        ], axis=1).dropna()
        aligned = aligned.loc[~aligned.index.duplicated(keep="last")]

        if aligned.empty or aligned["prev"].sum() == 0:
            res[q] = np.nan
        else:
            res[q] = (aligned["curr"].sum() - aligned["prev"].sum()) / aligned[
                "prev"].sum()

    return pd.Series(res).sort_index()
