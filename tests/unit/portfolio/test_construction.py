import numpy as np
import pandas as pd
import pytest

from pytrade.portfolio.construction import _numpy_vol_scale, _pandas_vol_scale, \
    _ignore_small_trades_numpy


@pytest.mark.parametrize(
    ["portfolio_weights", "asset_cov", "target_vol", "expected"],
    [
        pytest.param(
            np.array([
                [0.5, 0.5],  # vol=0.1
                [2.0, 2.0],  # vol=0.4
            ]),
            np.array([
                [[0.02, 0],
                 [0, 0.02]],
                [[0.02, 0],
                 [0, 0.02]],
            ]),
            0.2,
            np.array([
                [1.0, 1.0],
                [1.0, 1.0],
            ]),
            id="single_portfolio"
        ),
        pytest.param(
            np.array([
                [0.5, 0.5],  # vol=0.1
                [2.0, 2.0],  # vol=0.4
            ]),
            np.array([
                [[0.02, 0],
                 [0, 0.02]],
                [[0.02, 0],
                 [0, 0.02]],
            ]),
            np.array([0.2, 0.4]),
            np.array([
                [1.0, 1.0],
                [2.0, 2.0],
            ]),
            id="single_portfolio_dynamic_target_vol"
        ),
        pytest.param(
            np.array([
                [[0.5, 0.5],  # vol=0.1
                 [0.5, 0.5]],  # vol=0.1
                [[1.0, 1.0],  # vol=0.2
                 [2.0, 2.0]],  # vol=0.4
            ]),
            np.array([
                [[0.02, 0],
                 [0, 0.02]],
                [[0.02, 0],
                 [0, 0.02]],
            ]),
            0.2,
            np.array([
                [[1.0, 1.0],
                 [1.0, 1.0]],
                [[1.0, 1.0],
                 [1.0, 1.0]],
            ]),
            id="multiple_portfolios"
        ),
        pytest.param(
            np.array([
                [[0.5, 0.5],  # vol=0.1
                 [0.5, 0.5]],  # vol=0.1
                [[1.0, 1.0],  # vol=0.2
                 [2.0, 2.0]],  # vol=0.4
            ]),
            np.array([
                [[0.02, 0],
                 [0, 0.02]],
                [[0.02, 0],
                 [0, 0.02]],
            ]),
            np.array([0.2, 0.4]),
            np.array([
                [[1.0, 1.0],
                 [1.0, 1.0]],
                [[2.0, 2.0],
                 [2.0, 2.0]],
            ]),
            id="multiple_portfolios_dynamic_target_vol_1"
        ),
        pytest.param(
            np.array([
                [[0.5, 0.5],  # vol=0.1
                 [0.5, 0.5]],  # vol=0.1
                [[1.0, 1.0],  # vol=0.2
                 [2.0, 2.0]],  # vol=0.4
            ]),
            np.array([
                [[0.02, 0],
                 [0, 0.02]],
                [[0.02, 0],
                 [0, 0.02]],
            ]),
            np.array([
                [0.3, 0.2],
                [0.2, 0.4]
            ]),
            np.array([
                [[1.5, 1.5],
                 [1.0, 1.0]],
                [[1.0, 1.0],
                 [2.0, 2.0]],
            ]),
            id="multiple_portfolios_dynamic_target_vol_2"
        ),
    ]
)
def test__numpy_vol_scale(portfolio_weights, target_vol, asset_cov, expected):
    actual = _numpy_vol_scale(portfolio_weights, asset_cov, target_vol)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ["portfolio_weights", "asset_cov", "target_vol", "expected"],
    [
        # use same test cases as in test__numpy_vol_scale
        pytest.param(
            pd.DataFrame([
                [0.5, 0.5],  # vol=0.1
                [2.0, 2.0],  # vol=0.4
            ], columns=["A", "B"]),
            pd.DataFrame([
                [0.02, 0],
                [0, 0.02],
                [0.02, 0],
                [0, 0.02],
            ], index=pd.MultiIndex.from_product([range(2), ["A", "B"]]),
                columns=["A", "B"]),
            0.2,
            pd.DataFrame([
                [1.0, 1.0],
                [1.0, 1.0],
            ], columns=["A", "B"]),
            id="single_portfolio"
        ),
        pytest.param(
            pd.DataFrame([
                [0.5, 0.5],  # vol=0.1
                [0.5, 0.5],  # vol=0.1
                [1.0, 1.0],  # vol=0.2
                [2.0, 2.0],  # vol=0.4
            ], index=pd.MultiIndex.from_product([range(2), ["X", "Y"]]),
                columns=["A", "B"]),
            pd.DataFrame([
                [0.02, 0],
                [0, 0.02],
                [0.02, 0],
                [0, 0.02],
            ], index=pd.MultiIndex.from_product([range(2), ["A", "B"]]),
                columns=["A", "B"]),
            0.2,
            pd.DataFrame([
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ], index=pd.MultiIndex.from_product([range(2), ["X", "Y"]]),
                columns=["A", "B"]),
            id="multiple_portfolios"
        ),
    ]
)
def test__pandas_vol_scale(portfolio_weights, asset_cov, target_vol, expected):
    actual = _pandas_vol_scale(portfolio_weights, asset_cov, target_vol)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ["positions", "asset_prices", "min_trade_size", "expected"],
    [
        pytest.param(
            np.array([
                [[9.0, 4.2],
                 [10.0, 8.6]],
                [[2.1, 6.0],
                 [8.4, 7.6]],
                [[16.2, 12.1],
                 [11.5, 8.4]]
            ]),
            np.array([
                [1, 2],
                [1, 2],
                [1, 2]
            ]),
            2,
            np.array([
                [[9., 4.2],
                 [10., 8.6]],
                [[2.1, 6.],
                 [10., 7.6]],
                [[16.2, 12.1],
                 [10., 7.6]]
            ])
        ),
    ]
)
def test_ignore_small_trades_numpy(positions, asset_prices, min_trade_size, expected):
    actual = _ignore_small_trades_numpy(positions, asset_prices, min_trade_size)
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=0)
