import pandas as pd
import pytest
from pytrade.data import get_const_intervals
from pytrade.utils.pandas import str_to_pandas


@pytest.mark.parametrize(
    ["s", "expected"],
    [
        pytest.param(
            str_to_pandas("""
                index  v
                    0  0
                    1  0
                    2  0
                    3  1
                    4  2
                    5  3
                    6  3
                    7  4
      """, index_col="index", squeeze=True),
            str_to_pandas("""
            index  start  end  value
                0      0    2      0
                1      3    3      1
                2      4    4      2
                3      5    6      3
                4      7    7      4
          """, index_col="index"),
        ),
        pytest.param(
            str_to_pandas("""
                  index  v
                      0  0
                      1  0
                      2  0
                      3  1
                      4  2
                      5  3
                      6  3
                      7  3
                """, index_col="index", squeeze=True),
            str_to_pandas("""
              index  start  end  value
                  0      0    2      0
                  1      3    3      1
                  2      4    4      2
                  3      5    7      3
            """, index_col="index"),
        )
    ]
)
def test_get_const_intervals(s: pd.Series, expected: pd.DataFrame):
    actual = get_const_intervals(s)
    pd.testing.assert_frame_equal(actual, expected, check_names=False,
                                  atol=1e-3)
