import pandas as pd
import pytest

from pytrade.signal.masks import entry_exit_mask, EntryCondition
from pytrade.utils.pandas import str_to_pandas


@pytest.mark.parametrize(
    ["data", "entry_threshold", "exit_threshold", "entry_condition",
     "expected"],
    [
        pytest.param(
            str_to_pandas("""
                 time   a  b  c
                    0   1  8  5
                    1   2  5  5
                    2   5  3  5
                    3  10  2  5
                    4   7  7  7
                    5   6  6  6
                    6   4  6  4
                    7   8  2  7
            """, index_col="time"),
            7,
            4,
            EntryCondition.GREATER_THAN,
            str_to_pandas("""
                 time      a      b      c
                    0  False   True  False
                    1  False   True  False
                    2  False  False  False
                    3   True  False  False
                    4   True   True   True
                    5   True   True   True
                    6  False   True  False
                    7   True  False   True
              """, index_col="time"),
        ),
    ]
)
def test_entry_exit_mask(data, entry_threshold, exit_threshold,
                         entry_condition, expected):
    actual = entry_exit_mask(data, entry_threshold=entry_threshold,
                             exit_threshold=exit_threshold,
                             entry_condition=entry_condition)
    pd.testing.assert_frame_equal(actual, expected, check_names=False)
