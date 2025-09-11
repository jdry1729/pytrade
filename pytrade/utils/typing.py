from datetime import datetime, timedelta
from typing import Union, TypeVar
import pandas as pd

DatetimeOrFloat = Union[datetime, float]

TimedeltaOrFloat = Union[timedelta, float]

FloatOrSeries = Union[pd.Series, float]

T1 = TypeVar("T1", pd.DataFrame, pd.Series)
