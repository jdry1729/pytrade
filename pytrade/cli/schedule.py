from datetime import datetime
from typing import Optional, Sequence

import click


@click.command()
@click.argument("calendar", type=str)
@click.option("--start-time", type=click.DateTime(), default=None)
@click.option("--end-time", type=click.DateTime(), default=None)
@click.option("--trade-time", "-t", type=str, multiple=True)
@click.option("--start-index", type=int, default=None)
@click.option("--end-index", type=int, default=None)
@click.option("--format", "format_", type=str)
@click.option("--utc", is_flag=True, default=False)
def schedule(calendar: str, start_time: Optional[datetime] = None,
             end_time: Optional[datetime] = None,
             start_index: Optional[int] = None, end_index: Optional[int] = None,
             trade_time: Sequence[str] = (), format_: Optional[str] = None,
             utc: bool = False):
    from pytrade.utils.calendar import get_trade_schedule
    if format_ is None:
        format_ = "%Y-%m-%dT%H:%M:%S"

    trade_times = None
    if trade_time:
        trade_times = [datetime.strptime(x, "%H:%M").time() for x in trade_time]

    schedule_ = get_trade_schedule(calendar, start_time, end_time, trade_times, utc=utc)
    schedule_ = schedule_[slice(start_index, end_index)]
    for time in schedule_:
        print(time.strftime(format_))
