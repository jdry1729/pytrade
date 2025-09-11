import numpy as np
import pandas as pd
import pytest
from pytrade.utils.pandas import reindex_level, assert_equal, round_series


@pytest.mark.parametrize(
    ["obj", "index", "level", "method", "expected"],
    [
        pytest.param(
            pd.Series(range(9), index=pd.MultiIndex.from_product(
                [range(3), ["A", "B", "C"]])),
            range(5),
            0,
            None,
            pd.Series(list(range(9)) + [np.nan] * 6,
                      index=pd.MultiIndex.from_product(
                          [range(5), ["A", "B", "C"]])),
            id="dataframe"
        )
    ]
)
def test_reindex_level(obj, index, level, method, expected):
    actual = reindex_level(obj, index, level, method)
    assert_equal(actual, expected)


@pytest.mark.parametrize(
    ["s", "decimals", "expected"],
    [
        pytest.param(
            pd.Series([0.0001234, 0.01234, 0.1234]),
            3,
            pd.Series([0.0, 0.012, 0.123]),
        ),
        pytest.param(
            pd.Series({"A": 0.0001234, "B": 0.01234, "C": 0.1234}),
            pd.Series({"A": 5, "B": 3, "C": 2}),
            pd.Series({"A": 0.00012, "B": 0.012, "C": 0.12}),
        )
    ]
)
def test_round_series(s, decimals, expected):
    actual = round_series(s, decimals)
    pd.testing.assert_series_equal(actual, expected)
