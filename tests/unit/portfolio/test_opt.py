from typing import Union, Tuple, Optional

import numpy as np
import pytest
from pytrade.portfolio.opt import _numpy_single_period_markowitz_opt, \
    MarkowitzObj, _numpy_compute_single_period_char_portfolio
from pytrade.risk.utils import scale_vol_by_time


def _create_numpy_single_period_markowitz_opt_params(
        asset_cov: Union[np.ndarray, Tuple[
            np.ndarray, np.ndarray, np.ndarray]],
        *,
        objective: MarkowitzObj,
        expected: np.ndarray,
        asset_alphas: Optional[np.ndarray] = None,
        target_vol: Optional[float] = None,
        target_weights: Optional[np.ndarray] = None,
        init_weights: Optional[np.ndarray] = None,
        fixed_weights: Optional[np.ndarray] = None,
        max_leverage: Optional[float] = None,
        min_leverage: Optional[float] = None,
        max_pos_size: Optional[Union[float, np.ndarray]] = None,
        max_long_pos_size: Optional[Union[float, np.ndarray]] = None,
        max_short_pos_size: Optional[Union[float, np.ndarray]] = None,
        max_trade_size: Optional[Union[float, np.ndarray]] = None,
        long_only: bool = False,
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        asset_alphas_uncertainty_cov: Optional[np.ndarray] = None,
        asset_alphas_uncertainty_kappa: Optional[float] = None,
        id: Optional[str] = None
):
    return pytest.param(
        asset_cov,
        objective,
        asset_alphas,
        target_vol,
        target_weights,
        max_leverage,
        min_leverage,
        max_pos_size,
        max_long_pos_size,
        max_short_pos_size,
        long_only,
        linear_constraints,
        init_weights,
        fixed_weights,
        max_trade_size,
        asset_alphas_uncertainty_cov,
        asset_alphas_uncertainty_kappa,
        expected,
        id=id
    )


@pytest.mark.parametrize(
    ["asset_cov", "objective", "asset_alphas",
     "target_vol", "target_weights", "max_leverage", "min_leverage", "max_pos_size",
     "max_long_pos_size", "max_short_pos_size", "long_only", "linear_constraints",
     "init_weights", "fixed_weights", "max_trade_size", "asset_alphas_uncertainty_cov",
     "asset_alphas_uncertainty_kappa", "expected"],
    [
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0],
                      [0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.05, 0.05]),
            target_vol=0.1,
            expected=np.array([0.707, 0.707]),
            id="uncorrelated_assets"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0, 0, 0],
                      [0, 0.01, 0, 0],
                      [0, 0, 0.01, 0],
                      [0, 0, 0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.05, np.nan, 0.05, np.nan]),
            target_vol=0.1,
            expected=np.array([0.707, np.nan, 0.707, np.nan]),
            id="uncorrelated_assets_with_nans"
        ),
        # given a leverage cap as below, is there any reason why the [0.5, 0.5]
        # is chosen instead of [1, 0]?
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0],
                      [0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.05, 0.05]),
            target_vol=0.1,
            max_leverage=1,
            expected=np.array([0.5, 0.5]),
            id="uncorrelated_assets_with_leverage_constraint"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0],
                      [0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.05, 0.05]),
            target_vol=0.1,
            max_pos_size=0.1,
            expected=np.array([0.1, 0.1]),
            id="uncorrelated_assets_with_constant_max_pos_size"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0, 0, 0],
                      [0, 0.01, 0, 0],
                      [0, 0, 0.01, 0],
                      [0, 0, 0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.05, 0.05, 0.05, 0.05]),
            target_vol=0.05,
            max_pos_size=np.array([0.5, 0.2, 0.4, 0.05]),
            expected=np.array([0.322, 0.2, 0.322, 0.05]),
            id="uncorrelated_assets_with_variable_max_pos_size"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0, 0, 0],
                      [0, 0.01, 0, 0],
                      [0, 0, 0.01, 0],
                      [0, 0, 0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.05, 0.05, 0.05, 0.05]),
            target_vol=0.05,
            max_pos_size=np.array([0.5, np.nan, 0, 0.05]),
            expected=np.array([0.352, 0.352, 0.0, 0.05]),
            id="max_pos_size_contains_nan_and_zero"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0.005],
                      [0.005, 0.01]]),
            objective=MarkowitzObj.MIN_VARIANCE,
            min_leverage=1.0,
            long_only=True,
            expected=np.array([0.5, 0.5]),
            id="min_var_long_only_equal_asset_var"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.005, 0.002],
                      [0.002, 0.01]]),
            objective=MarkowitzObj.MIN_VARIANCE,
            min_leverage=1.0,
            long_only=True,
            expected=np.array([0.727, 0.273]),
            id="min_var_long_only_unequal_asset_var"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.005, 0.002],
                      [0.002, 0.01]]),
            objective=MarkowitzObj.MIN_TRACKING_ERROR,
            target_weights=np.array([0.5, 0.6]),
            expected=np.array([0.5, 0.6]),
            id="min_tracking_error_no_constraints"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.005, 0.002],
                      [0.002, 0.01]]),
            objective=MarkowitzObj.MIN_TRACKING_ERROR,
            target_weights=np.array([0.5, 0.6]),
            max_leverage=1.0,
            expected=np.array([0.427, 0.573]),
            id="min_tracking_error_max_leverage_constraint"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0],
                      [0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.05, 0.05]),
            target_vol=0.1,
            linear_constraints=(np.array([[1, 1]]), np.array([[np.nan, 0.5]])),
            expected=np.array([0.25, 0.25]),
            id="uncorrelated_assets_with_upper_net_leverage_constraint"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0],
                      [0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.05, 0.05]),
            target_vol=0.1,
            linear_constraints=(np.array([[1, -1]]), np.array([[0.2, np.nan]])),
            expected=np.array([0.8, 0.6]),
            id="uncorrelated_assets_with_lower_net_leverage_constraint"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0],
                      [0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.05, 0.05]),
            target_vol=0.1,
            linear_constraints=(np.array([[1, 0]]), np.array([[0, 0]])),
            expected=np.array([0, 1.0]),
            id="uncorrelated_assets_with_zero_position_constraint"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0],
                      [0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([-0.05, 0.05]),
            target_vol=0.1,
            max_long_pos_size=np.array([np.nan, 0.1]),
            max_short_pos_size=np.array([0.1, np.nan]),
            expected=np.array([-0.1, 0.1]),
            id="max_long_short_pos_size_constraints"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            (np.array([[1, 1, 0, 0],
                       [0, 0, 1, 1]]),
             np.array([[100, 50],
                       [25, 70]]),
             np.array([25, 25, 25, 25])),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([4, 4, 4, 4]),
            target_vol=20,
            expected=np.array([0.513, 0.513, 0.781, 0.781]),
            id="max_return_with_factor_model"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0],
                      [0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.0, 0.0]),
            target_vol=0.1,
            expected=np.array([np.nan, np.nan]),
            id="uncorrelated_assets_zero_returns"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.00039657, 0.00029949],
                      [0.00029949, 0.00037951]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.00313445, 0.00406203]),
            target_vol=scale_vol_by_time(0.3, 1 / 252),
            expected=np.array([-0.0402, 1.001])
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0],
                      [0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.05, 0.05]),
            target_vol=0.1,
            init_weights=np.array([0, 0.3]),
            max_trade_size=np.array([0, 0.1]),
            expected=np.array([0, 0.4]),
            id="max_trade_size_constraint"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0],
                      [0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.05, 0.05]),
            target_vol=0.1,
            init_weights=np.array([0, 0.3]),
            max_trade_size=np.array([0, np.nan]),
            expected=np.array([0, 1.0]),
            id="max_trade_size_constraint_with_nan"
        ),
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.01, 0],
                      [0, 0.01]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.05, 0.05]),
            target_vol=0.1,
            fixed_weights=np.array([0.5, np.nan]),
            expected=np.array([0.5, 0.866]),
            id="uncorrelated_assets_fixed_weight"
        ),
        # test case below from "A practical guide to robust portfolio optimization"
        # by Yin, Perchet and Soupé
        _create_numpy_single_period_markowitz_opt_params(
            np.array([[0.0366, 0.03945735, 0.00491907, 0.00842954],
                      [0.03945735, 0.0562, 0.00351665, 0.00704467],
                      [0.00491907, 0.00351665, 0.00978, 0.00942425],
                      [0.00842954, 0.00704467, 0.00942425, 0.0105]]),
            objective=MarkowitzObj.MAX_RETURN,
            asset_alphas=np.array([0.08800318, 0.10905008, 0.04549119, 0.04713597]),
            target_vol=0.1,
            asset_alphas_uncertainty_cov=np.array([[0.0366, 0., 0., 0.],
                                                   [0., 0.0562, 0., 0.],
                                                   [0., 0., 0.00978, 0.],
                                                   [0., 0., 0., 0.0105]]),
            asset_alphas_uncertainty_kappa=0.23,
            expected=np.array([0.149, 0.155, 0.377, 0.252])
        )
    ]
)
def test__numpy_single_period_markowitz_opt(
        asset_cov, objective, asset_alphas, target_vol, target_weights, max_leverage,
        min_leverage, max_pos_size, max_long_pos_size, max_short_pos_size, long_only,
        linear_constraints, init_weights, fixed_weights, max_trade_size,
        asset_alphas_uncertainty_cov, asset_alphas_uncertainty_kappa, expected):
    actual = _numpy_single_period_markowitz_opt(
        asset_cov=asset_cov,
        objective=objective,
        asset_alphas=asset_alphas,
        target_vol=target_vol,
        target_weights=target_weights,
        init_weights=init_weights,
        fixed_weights=fixed_weights,
        max_leverage=max_leverage,
        min_leverage=min_leverage,
        long_only=long_only,
        max_pos_size=max_pos_size,
        max_long_pos_size=max_long_pos_size,
        max_short_pos_size=max_short_pos_size,
        max_trade_size=max_trade_size,
        linear_constraints=linear_constraints,
        asset_alphas_uncertainty_cov=asset_alphas_uncertainty_cov,
        asset_alphas_uncertainty_kappa=asset_alphas_uncertainty_kappa)
    np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=0)


@pytest.mark.parametrize(["attribute", "asset_cov", "expected"], [
    pytest.param(
        np.array([0.02, 0.07]),
        np.array([[0.01, 0.01],
                  [0.01, 0.02]]),
        np.array([-10.34, 17.24]),
        id="two_assets"
    ),
    pytest.param(
        np.array([0.02, 0.07, np.nan]),
        np.array([[0.01, 0.01, np.nan],
                  [0.01, 0.02, np.nan],
                  [np.nan, np.nan, np.nan]]),
        np.array([-10.34, 17.24, np.nan]),
        id="nan_asset"
    ),
    pytest.param(
        np.array([0.02, 0.07, 0.03]),
        np.array([[0.01, 0.01, np.nan],
                  [0.01, 0.02, np.nan],
                  [np.nan, np.nan, np.nan]]),
        np.array([-10.34, 17.24, np.nan]),
        id="nan_asset"
    ),
    pytest.param(
        np.array([10, 20, 10, 20]),
        (
                np.array([[1, 1, 0, 0],
                          [0, 0, 1, 1]]),
                np.array([[100, 50],
                          [25, 70]]),
                np.array([25, 25, 25, 25])
        ),
        np.array([-0.023, 0.031, -0.016, 0.038]),
        id="structural_risk_model"
    )
])
def test__numpy_compute_single_period_characteristic_portfolio(
        attribute: np.ndarray, asset_cov: np.ndarray, expected: np.ndarray):
    actual = _numpy_compute_single_period_char_portfolio(
        attribute, asset_cov, unit_exposure=True)
    np.testing.assert_allclose(actual, expected, atol=1e-2)
