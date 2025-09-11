import pandas as pd
import pytest

from pytrade.portfolio.analysis import compute_portfolio_returns, \
    compute_weights_turnover, \
    compute_large_return_periods, get_funding_fees, compute_ib_commission
from pytrade.utils.pandas import str_to_pandas


@pytest.mark.parametrize(
    ["portfolio_weights", "asset_returns", "by_asset", "expected"],
    [
        pytest.param(
            str_to_pandas("""
      time   A    B    C
      0    0.1  0.2 -0.1  
      1    0.2  0.1    0
      2    0   -0.2  0.1
      """, index_col="time"),
            str_to_pandas("""
      time     A      B     C
      0     0.05  -0.02  0.01  
      1     0.02   0.01  0.02
      2    -0.01   0.01  0.03
      """, index_col="time"),
            False,
            str_to_pandas("""
      time   return
      0         nan 
      1       0.002
      2      -0.001
      """, index_col="time", squeeze=True),
            id="single_portfolio"
        ),
        pytest.param(
            str_to_pandas("""
        time portfolio    A    B    C
        0            0  0.1  0.2 -0.1
        0            1 -0.1 -0.2  0.1  
        1            0  0.2  0.1    0
        1            1 -0.2 -0.1    0
        2            0    0 -0.2  0.1
        2            1    0  0.2 -0.1
        """, index_col=["time", "portfolio"]),
            str_to_pandas("""
        time     A      B     C
        0     0.05  -0.02  0.01  
        1     0.02   0.01  0.02
        2    -0.01   0.01  0.03
        """, index_col="time"),
            False,
            str_to_pandas("""
        time   portfolio return
        0              0    nan
        0              1    nan 
        1              0  0.002
        1              1 -0.002
        2              0 -0.001
        2              1  0.001
        """, index_col=["time", "portfolio"], squeeze=True),
            id="many_portfolios"
        ),
        pytest.param(
            str_to_pandas("""
            time  portfolio   speed     A    B    C
            0             1       0   0.2  0.3  0.1
            0             1       1   0.1  0.2  0.3
            0             2       0   0.1  0.2  0.3
            0             2       1   0.1  0.1  0.1
            1             1       0  -0.1 -0.2 -0.3
            1             1       1  -0.1  0.1 -0.1
            1             2       0   0.2  0.3  0.1
            1             2       1   0.1  0.2  0.3
            2             1       0   0.1  0.2  0.3
            2             1       1   0.1  0.1  0.1
            2             2       0  -0.1 -0.2 -0.3
            2             2       1  -0.1  0.1 -0.1
            """, index_col=["time", "portfolio", "speed"]),
            str_to_pandas("""
            time    A     B     C
            0     0.1  0.02  0.03
            1    -0.2  0.10  0.10
            2    -0.1 -0.30  0.20
            """),
            False,
            str_to_pandas("""
            time portfolio speed return
            0            1     0   nan
            0            1     1   nan
            0            2     0   nan
            0            2     1   nan
            1            1     0   0.0
            1            1     1   0.03
            1            2     0   0.03
            1            2     1   0.0
            2            1     0   0.01
            2            1     1  -0.04
            2            2     0  -0.09
            2            2     1  -0.01
            """, index_col=["time", "portfolio", "speed"], squeeze=True),
            id="three_levels"
        ),
        # portfolio return should be nan when weights all nan
        pytest.param(
            str_to_pandas("""
        time   A    B    C
        0    0.1  0.2 -0.1  
        1    nan  nan  nan
        2    0   -0.2  0.1
        """, index_col="time"),
            str_to_pandas("""
        time     A      B     C
        0     0.05  -0.02  0.01  
        1     0.02   0.01  0.02
        2    -0.01   0.01  0.03
        """, index_col="time"),
            False,
            str_to_pandas("""
        time   return
        0         nan 
        1       0.002
        2         nan
        """, index_col="time", squeeze=True),
            id="single_portfolio_with_all_nan_row"
        ),
        pytest.param(
            str_to_pandas("""
                time   A    B    C
                0    0.1  0.2 -0.1  
                1    0.3  0.1 -0.2
                2    0   -0.2  0.1
                """, index_col="time"),
            str_to_pandas("""
                time     A      B     C
                0     0.05  -0.02  0.01  
                1     0.02   0.01  0.02
                2    -0.01   0.01  0.03
                """, index_col="time"),
            True,
            str_to_pandas("""
                time      A       B      C
                0     nan     nan      nan  
                1     0.002   0.002 -0.002
                2    -0.003   0.001 -0.006
                """, index_col="time"),
            id="single_portfolio_with_all_nan_row"
        ),
    ]
)
def test_compute_portfolio_returns(portfolio_weights, asset_returns, by_asset,
                                   expected):
    actual = compute_portfolio_returns(portfolio_weights, asset_returns,
                                       by_asset=by_asset)
    if isinstance(expected, pd.Series):
        return pd.testing.assert_series_equal(actual, expected, check_names=False)
    return pd.testing.assert_frame_equal(actual, expected, check_names=False)


@pytest.mark.parametrize(
    ["prices", "funding_rate", "expected"],
    [
        pytest.param(
            str_to_pandas("""
            time        PF_XBTUSD  PF_ETHUSD
            2024-02-23      36500      21000
            2024-02-24      37000      20000
            2024-02-25      39000      22000
            2024-02-26      40000      24000
            2024-02-27      42000      26000
            """, parse_dates=True, index_col="time"),
            str_to_pandas("""
            time                 PF_XBTUSD  PF_ETHUSD
            2024-02-24T00:00:00   0.274407   0.191721
            2024-02-24T06:00:00   0.357595   0.395863
            2024-02-24T12:00:00   0.301382   0.264447
            2024-02-24T18:00:00   0.272442   0.284022
            2024-02-25T00:00:00   0.211827   0.462798
            2024-02-25T06:00:00   0.322947   0.035518
            2024-02-25T12:00:00   0.218794   0.043565
            2024-02-25T18:00:00   0.445887   0.010109
            2024-02-26T00:00:00   0.481831   0.416310
            """, parse_dates=True, index_col="time"),
            str_to_pandas("""
            time        PF_XBTUSD  PF_ETHUSD
            2024-02-23   0.000000   0.000000
            2024-02-24   0.000000   0.000000
            2024-02-25   0.000033   0.000057
            2024-02-26   0.000031   0.000025
            2024-02-27   0.000012   0.000017
            """, parse_dates=True, index_col="time"),
            id="funding_rate_6h_prices_1D"
        ),
        pytest.param(
            str_to_pandas("""
            time                 PF_XBTUSD  PF_ETHUSD
            2024-02-23T23:30:00       3500       3400
            2024-02-24T00:00:00       3650       2100
            2024-02-24T00:30:00       3700       2000
            2024-02-24T01:00:00       3900       2200
            2024-02-24T01:30:00       4000       2400
            2024-02-24T02:00:00       4200       2600
            2024-02-24T02:30:00       4400       2500
            2024-02-24T03:00:00       4200       2100
            """, parse_dates=True, index_col="time"),
            str_to_pandas("""
            time                 PF_XBTUSD  PF_ETHUSD
            2024-02-24T00:00:00   0.274407   0.211827
            2024-02-24T01:00:00   0.357595   0.322947
            2024-02-24T02:00:00   0.301382   0.218794
            2024-02-24T03:00:00   0.272442   0.445887
            """, parse_dates=True, index_col="time"),
            str_to_pandas("""
            time                 PF_XBTUSD  PF_ETHUSD
            2024-02-23T23:30:00   0.000000   0.000000
            2024-02-24T00:00:00   0.000000   0.000000
            2024-02-24T00:30:00   0.000038   0.000050
            2024-02-24T01:00:00   0.000037   0.000053
            2024-02-24T01:30:00   0.000046   0.000073
            2024-02-24T02:00:00   0.000045   0.000067
            2024-02-24T02:30:00   0.000036   0.000042
            2024-02-24T03:00:00   0.000034   0.000044
            """, parse_dates=True, index_col="time"),
            id="funding_rate_1h_prices_30min"
        )
    ]
)
def test_get_funding_fees(prices, funding_rate, expected):
    actual = get_funding_fees(prices, funding_rate)
    pd.testing.assert_frame_equal(actual, expected, check_names=False,
                                  check_freq=False, atol=1e-6)


@pytest.mark.parametrize(
    ["trades", "asset_prices", "fee_per_share", "min_order_fee", "max_order_fee_pct",
     "expected"],
    [
        pytest.param(
            str_to_pandas("""
        time     A     B     C                     
        0     0.00  0.00  0.00
        1    -1.00 -0.33  0.00
        2    -0.33  2.33  0.33
        3    -2.16  0.60 -0.33
      """, index_col="time"),
            str_to_pandas("""
        time   A   B    C          
        0     10  20  100
        1     20  30   80
        2     30  40   60
        3     40  50   40
      """, index_col="time"),
            0.0035,
            0.35,
            0.01,
            str_to_pandas("""
            time  commission
            0       0.000
            1       0.299
            2       0.647
            3       0.782
            """, index_col="time", squeeze=True),
            id="single_portfolio"
        ),
    ]
)
def test_compute_ib_commission(trades, asset_prices, fee_per_share,
                               min_order_fee, max_order_fee_pct, expected):
    actual = compute_ib_commission(
        trades, asset_prices, fee_per_unit_trade=fee_per_share,
        min_trade_fee=min_order_fee, max_trade_fee_pct=max_order_fee_pct)
    pd.testing.assert_series_equal(actual, expected, check_names=False, atol=1e-6)


@pytest.mark.parametrize(
    ["portfolio_weights", "ann_factor", "expected"],
    [
        pytest.param(
            str_to_pandas("""
      time   A    B    C
      0    0.1  0.2 -0.1  
      1    0.2  0.1    0
      2    0   -0.2  0.1
      """, index_col="time"),
            1,
            str_to_pandas("""
      time   turnover
      0         nan 
      1         0.15
      2         0.3
      """, index_col="time", squeeze=True),
            id="single_portfolio"
        ),
    ]
)
def test_compute_weights_turnover(portfolio_weights, ann_factor, expected):
    actual = compute_weights_turnover(portfolio_weights, ann_factor=ann_factor)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


@pytest.mark.parametrize(
    ["returns", "window", "num_periods", "expected"],
    [
        pytest.param(
            pd.Series(
                [0.2, 0.3, 0.4, 0.1, -0.1, 0.2, -0.3, -0.4, -0.5, -0.2, 0.1,
                 0.1, 0.1]),
            3,
            5,
            str_to_pandas("""
       period       start       end  return
            0           6         8    -1.2
            1           0         2     0.9
            2          10        12     0.3
      """, index_col="period")
        )
    ]
)
def test_compute_large_return_periods(returns, window,
                                      num_periods, expected):
    actual = compute_large_return_periods(returns, window,
                                          num_periods=num_periods)
    pd.testing.assert_frame_equal(actual, expected, check_names=False)
