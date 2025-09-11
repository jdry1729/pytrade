import pandas as pd
import pytest

from pytrade.utils.pandas import str_to_pandas, stack


@pytest.mark.parametrize(
    ["objs", "keys", "names", "axis", "expected"],
    [
        pytest.param(
            [
                str_to_pandas("""
           time   A  B  C
              0   1  1  1
              1   2  2  2
      """, index_col="time"),
                str_to_pandas("""
           time    A   B  C
              0   -1  -1 -1
              1   -2  -2 -2
      """, index_col="time"),
            ],
            None,
            "portfolio",
            0,
            str_to_pandas(
                """
                time portfolio  A  B  C
                  0          0  1  1  1
                  0          1 -1 -1 -1
                  1          0  2  2  2
                  1          1 -2 -2 -2
                """, index_col=["time", "portfolio"]
            ),
            id="one_level"
        ),
        pytest.param(
            [
                str_to_pandas("""
                                 
                time portfolio  A  B  C        
                0            0  1  1  1
                0            1  2  2  2
                1            0  3  3  3
                1            1  4  4  4
            """, index_col=["time", "portfolio"]),
                str_to_pandas("""
                time portfolio   A   B   C        
                0            0  -1  -1  -1
                0            1  -2  -2  -2
                1            0  -3  -3  -3
                1            1  -4  -4  -4
            """, index_col=["time", "portfolio"]),
            ],
            [8, 12],
            "speed",
            0,
            str_to_pandas(
                """
                time portfolio speed   A   B  C
                   0    0        8      1  1  1
                   0    0        12    -1 -1 -1
                   0    1        8      2  2  2
                   0    1        12    -2 -2 -2
                   1    0        8      3  3  3
                   1    0        12    -3 -3 -3
                   1    1        8      4  4  4
                   1    1        12    -4 -4 -4
                """, index_col=["time", "portfolio", "speed"]
            ),
            id="two_levels"
        ),
        pytest.param(
            [
                str_to_pandas("""
                   a  b  c
                0  1  3  5
                1  2  4  6
            """),
                str_to_pandas("""
                   a  b  c
                0  1  3  5
                1  2  4  6
            """),
            ],
            ["m", "n"],
            None,
            1,
            str_to_pandas(
                """
                   a  a  b  b  c  c   
                   m  n  m  n  m  n
                   1  1  3  3  5  5
                   2  2  4  4  6  6
                """, header=[0, 1]),
            id="axis_1"
        ),
        pytest.param(
            {"a":
                 str_to_pandas("""
             time   A  B  C
                0   1  1  1
                1   2  2  2
        """, index_col="time"),
             "b":
                 str_to_pandas("""
             time    A   B  C
                0   -1  -1 -1
                1   -2  -2 -2
        """, index_col="time"),
             },
            None,
            "portfolio",
            0,
            str_to_pandas(
                """
                time portfolio  A  B  C
                  0          a  1  1  1
                  0          b -1 -1 -1
                  1          a  2  2  2
                  1          b -2 -2 -2
                """, index_col=["time", "portfolio"]
            ),
            id="objs_dict"
        ),
    ]
)
def test_stack(objs, keys, names, axis, expected):
    actual = stack(objs, keys=keys, names=names, axis=axis)
    pd.testing.assert_frame_equal(actual, expected)
