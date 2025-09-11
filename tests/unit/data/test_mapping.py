import pytest
import pandas as pd
from pytrade.data import map_ids
from pytrade.utils.pandas import str_to_pandas


@pytest.mark.parametrize(
    ["data", "mappings", "on", "expected"],
    [
        pytest.param(
            pd.DataFrame({
                "time": [0, 0, 0, 0, 2, 2, 2, 2, 5, 5, 5],
                "app": ["a", "b", "c", "d", "a", "b", "c", "d", "a", "b", "c"],
                "downloads": [10, 20, 30, 20, 10, 20, 30, 10, 20, 30, 20],
            }),
            pd.DataFrame({
                "app": ["a", "b", "c", "a"],
                "company": ["X", "X", "Z", "X"],
                "start_time": [0, 0, 0, 10],
                "end_time": [4, 4, 6, 12],
            }),
            "app",
            str_to_pandas("""
              time app  downloads company
                 0   a         10       X
                 0   b         20       X
                 0   c         30       Z
                 0   d         20     NaN
                 2   a         10       X
                 2   b         20       X
                 2   c         30       Z
                 2   d         10     NaN
                 5   a         20     NaN
                 5   b         30     NaN
                 5   c         20       Z
            """)
        )
    ])
def test__map_ids(data, mappings, on, expected):
    actual = map_ids(data, mappings, on)
    pd.testing.assert_frame_equal(actual, expected, check_names=False,
                                  atol=1e-3)
