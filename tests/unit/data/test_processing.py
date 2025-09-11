import pandas as pd
import pytest

from pytrade.data.processing import compute_xs_avg, merge_overlapping_intervals, \
    interval_difference
from pytrade.utils.pandas import str_to_pandas


@pytest.mark.parametrize(
    ["data", "weights", "expected"],
    [
        pytest.param(
            str_to_pandas("""
                index  A  B
                    0  0  1
                    1  2  3
                    2  4  5
      """, index_col="index"),
            None,
            str_to_pandas(
                """
                index average
                0         0.5
                1         2.5
                2         4.5
                """, index_col="index",
                squeeze=True
            ),
            id="unweighted"
        ),
        pytest.param(
            str_to_pandas("""
              index  A  B
                  0  0  1
                  1  2  3
                  2  4  5
            """, index_col="index"),
            {"A": 1, "B": 2},
            str_to_pandas(
                """
                index average
                0     0.666
                1     2.666
                2     4.666
                """, index_col="index",
                squeeze=True
            ),
            id="weighted"
        ),
    ]
)
def test_compute_xs_avg(data, weights, expected):
    actual = compute_xs_avg(data, weights)
    pd.testing.assert_series_equal(actual, expected, check_names=False, atol=1e-3)


@pytest.mark.parametrize(
    ["intervals", "expected"],
    [
        pytest.param(
            str_to_pandas("""
                index  start  end
                    0      1    6
                    1      4    8
                    2      2    3
                    3      -1   2
                    4      12  15
      """, index_col="index"),
            str_to_pandas("""
                index  start  end
                    0     -1    8
                    1     12   15
      """, index_col="index"),
        )
    ]
)
def test_merge_overlapping_intervals(intervals, expected):
    actual = merge_overlapping_intervals(intervals)
    pd.testing.assert_frame_equal(actual, expected, check_names=False)


@pytest.mark.parametrize(
    ["intervals_1", "intervals_2", "expected"],
    [
        pytest.param(
            str_to_pandas("""
            index  start  end
                0     -5   3
      """, index_col="index"),
            str_to_pandas("""
                index  start  end
                    0      1    6
                    1      4    8
                    2      2    3
                    3      -1   2
                    4      12  15
      """, index_col="index"),
            str_to_pandas("""
                index  start  end
                    0     -5   -1
      """, index_col="index"),
        ),
        pytest.param(
            str_to_pandas("""
                  index  start  end
                      0      6   20
            """, index_col="index"),
            str_to_pandas("""
              index  start  end
                  0      1    6
                  1      4    8
                  2      2    3
                  3      -1   2
                  4      12  15
            """, index_col="index"),
            str_to_pandas("""
              index  start  end
                  0      8   12
                  1     15   20
            """, index_col="index"),
        ),
        pytest.param(
            str_to_pandas("""
                  index  start  end
                      0      0    5
                      1     10   15
            """, index_col="index"),
            str_to_pandas("""
              index  start  end
                  0     -2    2
                  1      3    6
                  2     12   20
                  3     25   30
            """, index_col="index"),
            str_to_pandas("""
              index  start  end
                  0      2    3
                  1     10   12
            """, index_col="index"),
        )
    ]
)
def test_interval_difference(intervals_1, intervals_2, expected):
    actual = interval_difference(intervals_1, intervals_2)
    pd.testing.assert_frame_equal(actual, expected, check_names=False)
