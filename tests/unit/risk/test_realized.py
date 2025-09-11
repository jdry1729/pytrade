import pytest

from pytrade.risk.realized import compute_realized_cov
from pytrade.utils.pandas import str_to_pandas
import pandas as pd


@pytest.mark.parametrize(
    ["returns", "freq", "expected"],
    [
        pytest.param(
            str_to_pandas(
                """                
                time                        A         B            
                2024-02-20T00:00:00 -0.017224  0.020204
                2024-02-20T06:00:00  0.007553 -0.001298
                2024-02-20T12:00:00 -0.099728 -0.025464
                2024-02-20T18:00:00  0.092287  0.074630
                2024-02-21T00:00:00  0.038851  0.064523
                2024-02-21T06:00:00  0.037610 -0.035074
                2024-02-21T12:00:00  0.020732  0.025009
                2024-02-21T18:00:00 -0.029111 -0.068653
                2024-02-22T00:00:00 -0.001400 -0.082506
                2024-02-22T06:00:00  0.068362 -0.017188
                2024-02-22T12:00:00 -0.051465 -0.037321
                """,
                parse_dates=True,
                index_col="time"
            ),
            "1D",
            str_to_pandas(
                """  
                time      asset         A         B  
                2024-02-20    A       NaN       NaN
                2024-02-20    B       NaN       NaN
                2024-02-21    A  0.026705  0.015898
                2024-02-21    B  0.015898  0.013844
                2024-02-22    A  0.003592  0.001751
                2024-02-22    B  0.001751  0.017835
                2024-02-23    A  0.029288  0.002983
                2024-02-23    B  0.002983  0.006753
                """,
                index_col=("time", "asset"),
                parse_dates=["time"]
            )
        )
    ]
)
def test_compute_realized_cov(returns, freq, expected):
    actual = compute_realized_cov(returns, freq, min_sample_size=1)
    pd.testing.assert_frame_equal(actual, expected, atol=1e-6, rtol=0,
                                  check_names=False)
