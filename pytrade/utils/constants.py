import enum
from datetime import datetime

MIN_TIME = datetime(1900, 1, 1)
MAX_TIME = datetime(2100, 1, 1)


class Exchange(enum.Enum):
    BINANCE_SPOT = 0
    BINANCE_FUTURES_USDT = 1
    BINANCE_FUTURES_COIN = 2
