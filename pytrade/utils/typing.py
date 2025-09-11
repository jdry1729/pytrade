from datetime import datetime, timedelta
from typing import Union, TypeVar

import numpy as np
import pandas as pd

DatetimeOrFloat = Union[datetime, float]

TimedeltaOrFloat = Union[timedelta, float]

FloatOrSeries = Union[pd.Series, float]

T1 = TypeVar("T1", pd.DataFrame, pd.Series)
DataFrameOrArray = Union[pd.DataFrame, np.ndarray]
DataFrameOrSeries = Union[pd.DataFrame, pd.Series]
