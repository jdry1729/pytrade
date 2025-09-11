import enum

import pandas as pd

from pytrade.utils import stack


class Direction(enum.IntEnum):
    LONG = 1
    SHORT = -1


def positions_to_trades(positions: pd.DataFrame) -> pd.DataFrame:
    squeeze = False
    if positions.index.nlevels == 1:
        squeeze = True
        positions = stack([positions])

    levels = list(range(1, positions.index.nlevels))
    trades = positions.groupby(level=levels).diff().fillna(0)

    if squeeze:
        trades = trades.xs(0, level=1)

    return trades
