import numpy as np
import pandas as pd
import pytest

from pytrade.risk.models.cov import (_numpy_compute_full_asset_cov,
                                     _numpy_compute_portfolio_cov,
                                     _numpy_compute_factor_beta,
                                     _numpy_compute_portfolio_vol,
                                     _pandas_compute_factor_beta,
                                     _numpy_get_valid_assets)
from pytrade.stats.utils import _numpy_cov_to_corr


@pytest.mark.parametrize(
    ["portfolio_weights", "asset_cov", "expected"],
    [
        pytest.param(
            np.array([
                [[0.1, 0.1],
                 [0.1, 0.2]],
                [[0.1, 0.2],
                 [0.2, 0.2]],
            ]),
            np.array([
                [[0.01, 0.005],
                 [0.005, 0.01]],
                [[0.01, 0.005],
                 [0.005, 0.01]],
            ]),
            np.array([[0.017, 0.026],
                      [0.026, 0.034]]),
            id="multiple_portfolios",
        ),
        pytest.param(
            np.array([
                [0.1, 0.1],
                [0.1, 0.2],
            ]),
            np.array([
                [[0.01, 0.005],
                 [0.005, 0.01]],
                [[0.01, 0.005],
                 [0.005, 0.01]],
            ]),
            np.array([0.017, 0.027]),
            id="single_portfolio",
        ),
    ])
def test__numpy_portfolio_vol(portfolio_weights, asset_cov, expected):
    actual = _numpy_compute_portfolio_vol(portfolio_weights, asset_cov)
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=0)


@pytest.mark.parametrize(
    ["portfolio_weights", "factor_portfolio_weights", "asset_cov", "expected"],
    [
        pytest.param(
            np.array([
                [0.5, -0.6],
                [0.5, -0.6]
            ]),
            np.array([
                [[0.5, 0.5]],
                [[0.5, 0.5]],
            ]),
            np.array([
                [[0.09, 0.005],
                 [0.005, 0.09]],
                [[0.09, 0.3 * 0.5 * 0.5],
                 [0.3 * 0.5 * 0.5, 0.25]],
            ]),
            np.array([[-0.1],
                      [-0.46]]),
        ),
    ])
def test_numpy_compute_factor_beta(portfolio_weights, factor_portfolio_weights,
                                   asset_cov, expected):
    actual = _numpy_compute_factor_beta(portfolio_weights,
                                        factor_portfolio_weights, asset_cov)
    np.testing.assert_allclose(actual, expected, atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    ["portfolio_weights", "factor_portfolio_weights", "asset_cov", "expected"],
    [
        pytest.param(
            pd.DataFrame(np.array([
                [0.5, -0.6],
                [0.5, -0.6]
            ])),
            pd.DataFrame(np.array([
                [0.5, 0.5],
                [0.5, 0.5],
            ]), index=pd.RangeIndex(2)),
            pd.DataFrame(np.row_stack(np.array([
                [[0.09, 0.005],
                 [0.005, 0.09]],
                [[0.09, 0.3 * 0.5 * 0.5],
                 [0.3 * 0.5 * 0.5, 0.25]],
            ])), index=pd.MultiIndex.from_product((pd.RangeIndex(2), ("A", "B")))),
            pd.Series(np.array([-0.1, -0.46])),
        ),
    ])
def test_pandas_compute_factor_beta(portfolio_weights, factor_portfolio_weights,
                                    asset_cov, expected):
    actual = _pandas_compute_factor_beta(portfolio_weights,
                                         factor_portfolio_weights, asset_cov)
    pd.testing.assert_series_equal(actual, expected, atol=1e-2)


@pytest.mark.parametrize(["cov", "expected"], [
    pytest.param(
        np.array([
            [100, 25],
            [25, 25]
        ]),
        np.array([
            [1.0, 0.5],
            [0.5, 1.0],
        ]),
        id="2d"
    ),
    pytest.param(
        np.array([
            [[100, 25],
             [25, 25]],
            [[100, 25],
             [25, 25]],
        ]),
        np.array([
            [[1.0, 0.5],
             [0.5, 1.0]],
            [[1.0, 0.5],
             [0.5, 1.0]],
        ]),
        id="3d"
    )
])
def test__numpy_cov_to_corr(cov, expected):
    actual = _numpy_cov_to_corr(cov)
    np.testing.assert_allclose(actual, expected, atol=1e-2, rtol=0)


@pytest.mark.parametrize(["loadings", "factor_cov", "specific_var", "expected"], [
    pytest.param(
        np.array([
            [[1, 1, 0, 0],
             [0, 0, 1, 1]],
            [[1, 1, 0, 0],
             [0, 0, 1, 1]],
        ]),
        np.array([
            [[100, 25],
             [25, 25]],
            [[100, 25],
             [25, 25]]
        ]),
        np.zeros((2, 4)),
        np.array([
            [[100, 100, 25, 25],
             [100, 100, 25, 25],
             [25, 25, 25, 25],
             [25, 25, 25, 25]],

            [[100, 100, 25, 25],
             [100, 100, 25, 25],
             [25, 25, 25, 25],
             [25, 25, 25, 25]]
        ]),
        id="two_factors_zero_specific_var"
    ),
    pytest.param(
        np.array([
            [[1, 1, 0, 0],
             [0, 0, 1, 1]],
            [[1, 1, 0, 0],
             [0, 0, 1, 1]],
        ]),
        np.array([
            [[100, 25],
             [25, 25]],
            [[100, 25],
             [25, 25]]
        ]),
        np.array([[10, 0, 0, 10],
                  [10, 0, 0, 10]]),
        np.array([
            [[110, 100, 25, 25],
             [100, 100, 25, 25],
             [25, 25, 25, 25],
             [25, 25, 25, 35]],

            [[110, 100, 25, 25],
             [100, 100, 25, 25],
             [25, 25, 25, 25],
             [25, 25, 25, 35]]
        ]),
        id="two_factors_nonzero_specific_var"
    ),
    pytest.param(
        np.array([
            [[1, 1, 0, np.nan],
             [0, 0, 1, 1]],
            [[1, 1, 0, 0],
             [0, 0, np.nan, 1]],
        ]),
        np.array([
            [[100, 25],
             [25, 25]],
            [[100, 25],
             [25, 25]]
        ]),
        np.array([[10, 0, 0, 10],
                  [10, 0, 0, 10]]),
        np.array([
            [[110, 100, 25, np.nan],
             [100, 100, 25, np.nan],
             [25, 25, 25, np.nan],
             [np.nan, np.nan, np.nan, np.nan]],

            [[110, 100, np.nan, 25],
             [100, 100, np.nan, 25],
             [np.nan, np.nan, np.nan, np.nan],
             [25, 25, np.nan, 35]]
        ]),
        id="two_factors_nonzero_specific_var_with_nans"
    ),
])
def test__numpy_compute_full_asset_cov(loadings, factor_cov, specific_var, expected):
    actual = _numpy_compute_full_asset_cov(loadings, factor_cov, specific_var)
    np.testing.assert_allclose(actual, expected, atol=1e-2, rtol=0)


@pytest.mark.parametrize(
    ["portfolio_weights", "asset_cov", "expected"], [
        pytest.param(
            np.array([
                [[0.1, 0.1, 0.1, 0.1],
                 [-0.1, -0.1, -0.1, -0.1],
                 [0.1, 0.1, 0.1, 0.1]],
                [[0.2, 0.2, 0.2, 0.2],
                 [-0.2, -0.2, -0.2, -0.2],
                 [0.2, 0.2, 0.2, 0.2]],
            ]),
            (
                    np.array([
                        [[1, 1, 0, 0],
                         [0, 0, 1, 1]],
                        [[1, 1, 0, 0],
                         [0, 0, 1, 1]],
                    ]),
                    np.array([
                        [[100, 25],
                         [25, 25]],
                        [[100, 25],
                         [25, 25]]
                    ]),
                    np.array([[50, 0, 50, 0],
                              [50, 0, 50, 0]])
            ),
            np.array(
                [[[8, -8, 8],
                  [-8, 8, -8],
                  [8, -8, 8]],

                 [[32, -32, 32],
                  [-32, 32, -32],
                  [32, -32, 32]]]
            ),
            id="two_factors_nonzero_specific_var"
        ),
        pytest.param(
            np.array([
                [[0.1, 0.1],
                 [0.1, 0.2]],
                [[0.1, 0.2],
                 [0.2, 0.2]],
            ]),
            np.array([
                [[0.01, 0.005],
                 [0.005, 0.01]],
                [[0.01, 0.005],
                 [0.005, 0.01]],
            ]),
            np.array(
                [[[0.0003, 0.00045],
                  [0.00045, 0.0007]],
                 [[0.0007, 0.0009],
                  [0.0009, 0.0012]]]
            ),
            id="full_asset_cov"
        ),
    ])
def test__numpy_compute_portfolio_cov(portfolio_weights, asset_cov, expected):
    actual = _numpy_compute_portfolio_cov(portfolio_weights, asset_cov)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ["portfolio_weights", "asset_cov", "expected"],
    [
        pytest.param(
            np.array([
                [[0.1, 0.1],
                 [0.1, 0.2]],
                [[0.1, 0.2],
                 [0.2, 0.2]],
            ]),
            np.array([
                [[0.01, 0.005],
                 [0.005, 0.01]],
                [[0.01, 0.005],
                 [0.005, 0.01]],
            ]),
            np.array(
                [[[0.0003, 0.00045],
                  [0.00045, 0.0007]],
                 [[0.0007, 0.0009],
                  [0.0009, 0.0012]]])
        )
    ]
)
def test__numpy_portfolio_cov(portfolio_weights, asset_cov, expected):
    actual = _numpy_compute_portfolio_cov(portfolio_weights, asset_cov)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ["loadings", "factor_cov", "specific_var", "expected"],
    [
        pytest.param(
            np.array(
                [
                    [[np.nan, 1, 0, 0],
                     [0, 0, 1, 1]],
                    [[1, 1, 0, 0],
                     [0, 0, 1, 1]],
                    [[1, 1, 0, 0],
                     [0, 1, 1, 1]],
                ]
            ),
            np.array([[[100, 25],
                       [25, 25]],
                      [[100, 25],
                       [25, np.nan]],
                      [[np.nan, 25],
                       [25, np.nan]]]),
            np.zeros((3, 4)),
            np.array([[False, True, True, True],
                      [True, True, False, False],
                      [False, False, False, False]]),
            id="multi_period_structural_asset_cov"
        ),
        pytest.param(
            np.array([[np.nan, 1, 0, 0],
                      [0, 0, 1, 1]]),
            np.array([[100, 25], [25, 25]]),
            np.zeros(4),
            np.array([False,  True,  True,  True]),
            id="single_period_structural_asset_cov"
        )
    ]
)
def test__numpy_get_valid_assets(loadings, factor_cov, specific_var, expected):
    mask = _numpy_get_valid_assets(loadings, factor_cov, specific_var)
    np.testing.assert_allclose(mask, expected)
