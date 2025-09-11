from __future__ import annotations

import enum
import logging
import operator
from typing import Optional, Union, Tuple, List

import cvxpy as cp
import numpy as np
import pandas as pd
from cvxpy.atoms.affine.wraps import psd_wrap
from pytrade.risk.models.cov import _numpy_get_valid_assets, \
    _numpy_compute_single_period_full_asset_cov
from pytrade.utils.collections import is_iterable_of, all_none
from pytrade.utils.pandas import pandas_to_numpy, raise_if_index_not_equal
from tqdm import tqdm

logger = logging.getLogger(__name__)

BIG_M = 1e6
EPS = 1e-6


class MarkowitzObj(enum.Enum):
    MAX_RETURN = 0
    MIN_VARIANCE = 1
    MIN_TRACKING_ERROR = 2


def _numpy_compute_single_period_char_portfolio(
        attribute: np.ndarray,
        asset_cov: Union[np.ndarray, Tuple[
            np.ndarray, np.ndarray, np.ndarray]],
        unit_exposure: bool = False,
) -> np.ndarray:
    if isinstance(asset_cov, tuple):
        # simplest way to handle factor cov input is just to expand it into full
        # asset cov first
        asset_cov = _numpy_compute_single_period_full_asset_cov(
            asset_cov[0], asset_cov[1], asset_cov[2])

    N = asset_cov.shape[0]
    opt_weights = np.full(N, np.nan)

    # we assume that if variance of returns of two assets exists, then their
    # covariance also exists
    invalid_mask = np.isnan(attribute) | np.isnan(np.diag(asset_cov))
    asset_cov = asset_cov[np.ix_(~invalid_mask, ~invalid_mask)]
    attribute = attribute[~invalid_mask]

    weights = np.linalg.solve(asset_cov, attribute)
    if unit_exposure:
        exposure = attribute.T @ weights
        weights /= exposure

    opt_weights[~invalid_mask] = weights
    return opt_weights


def _pandas_compute_single_period_char_portfolio(
        attribute: pd.Series,
        asset_cov: Union[pd.DataFrame, Tuple[
            pd.DataFrame, pd.DataFrame, pd.Series]],
        unit_exposure: bool = False,
) -> pd.Series:
    if isinstance(asset_cov, pd.DataFrame):
        assets = asset_cov.columns
    else:
        assets = asset_cov[2].index

    opt_weights = _numpy_compute_single_period_char_portfolio(
        pandas_to_numpy(attribute), pandas_to_numpy(asset_cov),
        unit_exposure=unit_exposure
    )
    return pd.Series(opt_weights, index=assets)


def compute_single_period_char_portfolio(attribute, asset_cov,
                                         unit_exposure: bool = False):
    kwargs = {
        "attribute": attribute,
        "asset_cov": asset_cov,
        "unit_exposure": unit_exposure,
    }
    if isinstance(asset_cov, pd.DataFrame) or is_iterable_of(
            asset_cov, (pd.DataFrame, pd.Series)):
        return _pandas_compute_single_period_char_portfolio(**kwargs)
    elif isinstance(asset_cov, np.ndarray) or is_iterable_of(asset_cov, np.ndarray):
        return _numpy_single_period_markowitz_opt(**kwargs)
    raise ValueError("asset_cov must be a array-like or a tuple of array-like objects")


def _numpy_compute_char_portfolio(
        attribute: np.ndarray,
        asset_cov: Union[np.ndarray, Tuple[
            np.ndarray, np.ndarray, np.ndarray]],
        unit_exposure: bool = False) -> np.ndarray:
    """
    Computes characteristic portfolio for an attribute over time.
    """
    # it'd be faster to use np.solve to compute char portfolio for each timestep
    # in single call; however, since this function is very fast in its current form,
    # it's not important to make this change
    if isinstance(asset_cov, np.ndarray):
        T = asset_cov.shape[0]
    else:
        T = asset_cov[2].shape[0]

    res = []
    for i in range(T):

        if isinstance(asset_cov, np.ndarray):
            asset_cov_ = asset_cov[i]
        else:
            asset_cov_ = tuple(x[i] for x in asset_cov)

        opt_weights = _numpy_compute_single_period_char_portfolio(
            attribute[i],
            asset_cov_,
            unit_exposure=unit_exposure
        )
        res.append(opt_weights)
    return np.vstack(res)


def _pandas_compute_char_portfolio(attribute: pd.DataFrame,
                                   asset_cov: Union[pd.DataFrame, Tuple[
                                       pd.DataFrame, pd.DataFrame, pd.DataFrame]],
                                   unit_exposure: bool = False) -> pd.DataFrame:
    if isinstance(asset_cov, pd.DataFrame):
        times = asset_cov.index.unique(level=0)
        assets = asset_cov.columns
    else:
        times = asset_cov[2].index
        assets = asset_cov[2].columns

    opt_weights = _numpy_compute_char_portfolio(
        attribute=attribute.values,
        asset_cov=pandas_to_numpy(asset_cov),
        unit_exposure=unit_exposure)
    return pd.DataFrame(opt_weights, index=times, columns=assets)


def compute_char_portfolio(attribute, asset_cov, unit_exposure: bool = False):
    """
    Computes characteristic portfolio of an attribute.

    Parameters
    ----------
    attribute
        Attribute. If attribute for an asset is NaN, its weight is
        constrained to be 0.
    asset_cov
        Asset covariance matrix. Must not contain NaNs.
    unit_exposure
        Whether to scale the exposure of the portfolio to the attribute to 1.

    Returns
    -------
    Opt weights.
    """
    array_like = [attribute]
    if is_iterable_of(array_like, pd.DataFrame):
        return _pandas_compute_char_portfolio(attribute, asset_cov,
                                              unit_exposure)
    elif is_iterable_of(array_like, np.ndarray):
        return _numpy_compute_char_portfolio(attribute, asset_cov,
                                             unit_exposure)
    raise ValueError("attribute must be a dataframe or numpy array")


def markowitz_opt(asset_cov,
                  objective=MarkowitzObj.MAX_RETURN,
                  *,
                  asset_alphas=None,
                  asset_returns=None,
                  target_vol=None,
                  target_weights=None,
                  target_return=None,
                  init_weights=None,
                  fixed_weights=None,
                  max_leverage=None,
                  min_leverage=None,
                  max_pos_size=None,
                  min_pos_size=1e-4,
                  max_long_pos_size=None,
                  max_short_pos_size=None,
                  max_trade_size=None,
                  soft_max_trade_size=None,
                  soft_max_trade_size_penalty=None,
                  buy_veto_mask=None,
                  sell_veto_mask=None,
                  long_only=False,
                  allow_pos_flip=True,
                  l1_weights_penalty=None,
                  l1_trades_penalty=None,
                  linear_constraints=None,
                  min_trade_size: Optional[float] = None,
                  max_trades: Optional[int] = None,
                  max_total_trade_size: Optional[float] = None,
                  max_positions: Optional[int] = None,
                  asset_alphas_uncertainty_cov: Optional = None,
                  asset_alphas_uncertainty_kappa: Optional = None,
                  trades_fee: Optional = None,
                  soft_max_total_trades_fee=None,
                  soft_max_total_trades_fee_penalty=None,
                  solver: Optional[str] = None,
                  show_progress: bool = True,
                  **kwargs,
                  ):
    kwargs_ = {
        "asset_cov": asset_cov,
        "objective": objective,
        "asset_alphas": asset_alphas,
        "asset_returns": asset_returns,
        "target_vol": target_vol,
        "target_weights": target_weights,
        "target_return": target_return,
        "init_weights": init_weights,
        "fixed_weights": fixed_weights,
        "max_leverage": max_leverage,
        "min_leverage": min_leverage,
        "max_pos_size": max_pos_size,
        "min_pos_size": min_pos_size,
        "max_long_pos_size": max_long_pos_size,
        "max_short_pos_size": max_short_pos_size,
        "max_trade_size": max_trade_size,
        "soft_max_trade_size": soft_max_trade_size,
        "soft_max_trade_size_penalty": soft_max_trade_size_penalty,
        "buy_veto_mask": buy_veto_mask,
        "sell_veto_mask": sell_veto_mask,
        "long_only": long_only,
        "allow_pos_flip": allow_pos_flip,
        "l1_weights_penalty": l1_weights_penalty,
        "l1_trades_penalty": l1_trades_penalty,
        "linear_constraints": linear_constraints,
        "min_trade_size": min_trade_size,
        "max_trades": max_trades,
        "max_total_trade_size": max_total_trade_size,
        "max_positions": max_positions,
        "asset_alphas_uncertainty_cov": asset_alphas_uncertainty_cov,
        "asset_alphas_uncertainty_kappa": asset_alphas_uncertainty_kappa,
        "trades_fee": trades_fee,
        "soft_max_total_trades_fee": soft_max_total_trades_fee,
        "soft_max_total_trades_fee_penalty": soft_max_total_trades_fee_penalty,
        "solver": solver,
        "show_progress": show_progress,
    }
    if isinstance(asset_cov, pd.DataFrame) or is_iterable_of(asset_cov, pd.DataFrame):
        return _pandas_markowitz_opt(**kwargs_, **kwargs)
    elif isinstance(asset_cov, np.ndarray) or is_iterable_of(asset_cov, np.ndarray):
        return _numpy_markowitz_opt(**kwargs_, **kwargs)
    raise ValueError(
        "asset_cov must be a array-like or a tuple of array-like objects")


def _pandas_markowitz_opt(
        asset_cov: Union[pd.DataFrame, Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame]],
        objective: MarkowitzObj = MarkowitzObj.MAX_RETURN,
        *,
        asset_alphas: Optional[pd.DataFrame] = None,
        asset_returns: Optional[pd.DataFrame] = None,
        target_vol: Optional[Union[float, pd.Series]] = None,
        target_weights: Optional[pd.DataFrame] = None,
        target_return: Optional[float] = None,
        init_weights: Optional[pd.Series] = None,
        fixed_weights: Optional[pd.DataFrame] = None,
        max_leverage: Optional[float] = None,
        min_leverage: Optional[float] = None,
        max_pos_size: Optional[Union[float, pd.DataFrame]] = None,
        min_pos_size: Optional[float] = 1e-4,
        max_long_pos_size: Optional[Union[float, pd.DataFrame]] = None,
        max_short_pos_size: Optional[Union[float, pd.DataFrame]] = None,
        max_trade_size: Optional[Union[float, pd.DataFrame]] = None,
        soft_max_trade_size: Optional[Union[float, pd.DataFrame]] = None,
        soft_max_trade_size_penalty: Optional[float] = None,
        buy_veto_mask: Optional[pd.DataFrame] = None,
        sell_veto_mask: Optional[pd.DataFrame] = None,
        long_only: bool = False,
        allow_pos_flip: bool = True,
        l1_weights_penalty: Optional[float] = None,
        l1_trades_penalty: Optional[float] = None,
        linear_constraints: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
        min_trade_size: Optional[float] = None,
        max_trades: Optional[int] = None,
        max_total_trade_size: Optional[float] = None,
        max_positions: Optional[int] = None,
        asset_alphas_uncertainty_cov: Optional[pd.DataFrame] = None,
        asset_alphas_uncertainty_kappa: Optional[Union[float, pd.Series]] = None,
        trades_fee: Optional[Union[float, pd.DataFrame]] = None,
        soft_max_total_trades_fee: Optional[Union[float, pd.Series]] = None,
        soft_max_total_trades_fee_penalty: Optional[float] = None,
        solver: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
) -> pd.DataFrame:
    """
    Computes optimal portfolio weights given an objective and constraints.
    """
    if isinstance(asset_cov, pd.DataFrame):
        times = asset_cov.index.unique(level=0)
        assets = asset_cov.columns
    else:
        times = asset_cov[2].index
        assets = asset_cov[2].columns

    to_check = {"asset_alphas": asset_alphas,
                "asset_returns": asset_returns,
                "target_weights": target_weights,
                "fixed_weights": fixed_weights,
                "target_vol": target_vol,
                "max_pos_size": max_pos_size,
                "soft_max_trade_size": soft_max_trade_size,
                "max_long_pos_size": max_long_pos_size,
                "max_short_pos_size": max_short_pos_size,
                "max_trade_size": max_trade_size,
                "buy_veto_mask": buy_veto_mask,
                "sell_veto_mask": sell_veto_mask,
                "trades_fee": trades_fee,
                }
    for key in to_check:
        obj = to_check[key]
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            try:
                raise_if_index_not_equal(to_check[key].index, times)
            except ValueError:
                raise ValueError(f"Error computing opt weights; asset_cov and {key}"
                                 " must have same time index")

    if linear_constraints is not None:
        # TODO: extend pandas_to_numpy to allow multiple arrays to be returned
        #  according to group by statement
        # linear constraints must be subset of asset_cov times
        constraint_times = linear_constraints[0].index.unique(level=0)
        constraint_ilocs = times.get_indexer(constraint_times).tolist()
        constraint_mats = linear_constraints[0].groupby(level=0).apply(
            lambda x: x.values).tolist()
        bounds = linear_constraints[1].groupby(level=0).apply(
            lambda x: x.values).tolist()
        linear_constraints = list(zip(constraint_ilocs, constraint_mats, bounds))

    opt_weights = _numpy_markowitz_opt(
        asset_cov=pandas_to_numpy(asset_cov),
        objective=objective,
        asset_alphas=pandas_to_numpy(asset_alphas),
        asset_returns=pandas_to_numpy(asset_returns),
        target_vol=pandas_to_numpy(target_vol),
        target_weights=pandas_to_numpy(target_weights),
        target_return=pandas_to_numpy(target_return),
        init_weights=pandas_to_numpy(init_weights),
        fixed_weights=pandas_to_numpy(fixed_weights),
        max_leverage=max_leverage,
        min_leverage=min_leverage,
        max_pos_size=pandas_to_numpy(max_pos_size),
        min_pos_size=pandas_to_numpy(min_pos_size),
        max_long_pos_size=pandas_to_numpy(max_long_pos_size),
        max_short_pos_size=pandas_to_numpy(max_short_pos_size),
        max_trade_size=pandas_to_numpy(max_trade_size),
        soft_max_trade_size=pandas_to_numpy(soft_max_trade_size),
        soft_max_trade_size_penalty=soft_max_trade_size_penalty,
        buy_veto_mask=pandas_to_numpy(buy_veto_mask),
        sell_veto_mask=pandas_to_numpy(sell_veto_mask),
        long_only=long_only,
        allow_pos_flip=allow_pos_flip,
        l1_weights_penalty=l1_weights_penalty,
        l1_trades_penalty=l1_trades_penalty,
        linear_constraints=linear_constraints,
        min_trade_size=min_trade_size,
        max_trades=max_trades,
        max_total_trade_size=max_total_trade_size,
        max_positions=max_positions,
        asset_alphas_uncertainty_cov=pandas_to_numpy(asset_alphas_uncertainty_cov),
        asset_alphas_uncertainty_kappa=pandas_to_numpy(asset_alphas_uncertainty_kappa),
        trades_fee=pandas_to_numpy(trades_fee),
        soft_max_total_trades_fee=pandas_to_numpy(soft_max_total_trades_fee),
        soft_max_total_trades_fee_penalty=soft_max_total_trades_fee_penalty,
        solver=solver,
        show_progress=show_progress,
        **kwargs,
    )

    return pd.DataFrame(opt_weights, index=times, columns=assets)


def _numpy_markowitz_opt(
        asset_cov: Union[np.ndarray, Tuple[
            np.ndarray, np.ndarray, np.ndarray]],
        objective: MarkowitzObj = MarkowitzObj.MAX_RETURN,
        *,
        asset_alphas: Optional[np.ndarray] = None,
        asset_returns: Optional[np.ndarray] = None,
        target_vol: Optional[Union[float, np.ndarray]] = None,
        target_weights: Optional[np.ndarray] = None,
        target_return: Optional[float] = None,
        init_weights: Optional[np.ndarray] = None,
        fixed_weights: Optional[np.ndarray] = None,
        max_leverage: Optional[float] = None,
        min_leverage: Optional[float] = None,
        max_pos_size: Optional[Union[float, np.ndarray]] = None,
        min_pos_size: Optional[float] = 1e-4,
        max_long_pos_size: Optional[Union[float, np.ndarray]] = None,
        max_short_pos_size: Optional[Union[float, np.ndarray]] = None,
        max_trade_size: Optional[Union[float, np.ndarray]] = None,
        soft_max_trade_size: Optional[Union[float, np.ndarray]] = None,
        soft_max_trade_size_penalty: Optional[float] = None,
        buy_veto_mask: Optional[np.ndarray] = None,
        sell_veto_mask: Optional[np.ndarray] = None,
        long_only: bool = False,
        allow_pos_flip: bool = True,
        l1_weights_penalty: Optional[float] = None,
        l1_trades_penalty: Optional[float] = None,
        linear_constraints: Optional[List[Tuple[int, np.ndarray, np.ndarray]]] = None,
        min_trade_size: Optional[float] = None,
        max_trades: Optional[int] = None,
        max_total_trade_size: Optional[float] = None,
        max_positions: Optional[int] = None,
        asset_alphas_uncertainty_cov: Optional[np.ndarray] = None,
        asset_alphas_uncertainty_kappa: Optional[Union[float, np.ndarray]] = None,
        trades_fee: Optional[Union[float, np.ndarray]] = None,
        soft_max_total_trades_fee: Optional[Union[float, np.ndarray]] = None,
        soft_max_total_trades_fee_penalty: Optional[float] = None,
        solver: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
) -> np.ndarray:
    """
    Computes Markowitz optimal portfolio weights.

    Parameters
    ----------
    asset_cov
        Either a full asset cov matrix with shape (T, N, N), or a tuple of form
        (loadings, factor_cov, specific_var). Loadings must have shape (T, K, N).
        Factor covariance must have shape (T, K, K) and specific var must have
        shape (T, N).
    asset_alphas: (T, N)
        Expected asset returns.
    asset_returns
        Asset returns. Must be specified if max_trades or l1_trade_penalty set.
    target_vol
        Target volatility. May be float if constant, or array if dynamic.
    target_weights
        Portfolio to minimize tracking error to. Can only be used with
        MIN_TRACKING_ERROR objective.
    init_weights
        Initial portfolio weights. These weights will be passed as the initial weights
        for the optimization at each time step until the weights output by the
        optimization contain at least one non-nan value.
    fixed_weights
        Fixed weights to hold.
    max_leverage
        Maximum leverage.
    max_pos_size
        Maximum position size. Can be an array to set different max pos size
        for each asset.
    max_long_pos_size
        Maximum long position size. Can be an array to set different value for each
        asset.
    max_short_pos_size
        Maximum short position size. Can be an array to set different value for each
        asset.
    max_trade_size
        Maximum trade size. Can be an array to set different value for each asset.
    soft_max_trade_size
        Soft maximum trade size. Can be an array to set different value for each asset.
    soft_max_trade_size_penalty
        Soft maximum trade size penalty.
    long_only
        Whether all weights must be positive.
    allow_pos_flip
        Controls whether weight is allowed to have opposite sign to
        expected return.
    l1_weights_penalty
        L1 weights penalty. A higher value will result in sparser portfolios,
        i.e., portfolios with fewer active positions.
    l1_trades_penalty
        L1 trades penalty. A higher value will result in fewer trades.
    linear_constraints
        Linear constraints to apply to the optimal portfolio weights. Must be a list
        of tuples. The first element of each tuple contains the index to which the
        constraint applies, the second is the constraint matrix, with shape K x N,
        and third is a K x 2 matrix of lower and upper bounds.
    min_trade_size
        Minimum trade size as a proportion of FUM. Optimization will take considerably
        longer if set since requires mixed-integer solver.
    max_trades
        Maximum number of trades at each timestep.
    max_total_trade_size
        Maximum total trade size as a proportion of FUM.
    max_positions
        Maximum number of positions at each timestep.
    trades_fee
        Fee as percent of notional trade value. Can be used to model commission,
        bid/ask spread cost, etc.
    solver
        Solver to use. If not set, CVXPY will choose whichever is most appropriate.
    kwargs
        Additional kwargs to pass to solver.
    show_progress
        Whether to show optimization progress.

    Returns
    -------
    out : (T, N)
        Optimal portfolio weights.
    """
    if isinstance(asset_cov, np.ndarray):
        T = asset_cov.shape[0]
        N = asset_cov.shape[1]
    else:
        T, N = asset_cov[2].shape

    use_init_weights = not all_none((max_trades, max_total_trade_size,
                                     l1_trades_penalty, min_trade_size,
                                     trades_fee))
    if use_init_weights:
        if asset_returns is None:
            raise ValueError("Error computing Markowitz optimal weights; asset_returns"
                             " must be specified if trades constraint/ penalty is")
        if init_weights is None:
            init_weights = np.zeros(N)
    else:
        init_weights = None

    if linear_constraints is not None:
        # copy constraints to avoid mutating original
        linear_constraints = linear_constraints.copy()

    # TODO: could create loc function for this?
    def _get(obj: Optional[Union[np.ndarray, float]], i: int) -> Optional[
        Union[np.ndarray, float]]:
        if isinstance(obj, np.ndarray):
            return obj[i]
        return obj

    res = []
    # portfolio is "valid" once at least one asset has a non-nan weight
    is_valid = False
    opt_weights = None
    for i in tqdm(range(T), disable=(not show_progress)):
        if is_valid and use_init_weights:
            # must fillna below - otherwise weight for valid asset at time T will
            # be nan if it was invalid at T - 1
            # adjust weights to account for price changes from T to T + 1
            init_weights = np.nan_to_num(opt_weights) * (1.0 + asset_returns[i])

        if isinstance(asset_cov, np.ndarray):
            asset_cov_ = asset_cov[i]
        else:
            asset_cov_ = tuple(x[i] for x in asset_cov)

        linear_constraints_ = None
        if linear_constraints is not None and linear_constraints:
            # check whether time index of first linear constraint in queue matches
            # current time index
            if linear_constraints[0][0] == i:
                linear_constraints_ = linear_constraints.pop(0)[1:]

        opt_weights = _numpy_single_period_markowitz_opt(
            asset_cov=asset_cov_,
            objective=objective,
            asset_alphas=_get(asset_alphas, i),
            target_vol=_get(target_vol, i),
            target_weights=_get(target_weights, i),
            target_return=target_return,
            init_weights=init_weights,
            fixed_weights=_get(fixed_weights, i),
            max_leverage=max_leverage,
            min_leverage=min_leverage,
            max_pos_size=_get(max_pos_size, i),
            min_pos_size=_get(min_pos_size, i),
            max_long_pos_size=_get(max_long_pos_size, i),
            max_short_pos_size=_get(max_short_pos_size, i),
            max_trade_size=_get(max_trade_size, i),
            soft_max_trade_size=_get(soft_max_trade_size, i),
            soft_max_trade_size_penalty=soft_max_trade_size_penalty,
            buy_veto_mask=_get(buy_veto_mask, i),
            sell_veto_mask=_get(sell_veto_mask, i),
            long_only=long_only,
            allow_pos_flip=allow_pos_flip,
            l1_weights_penalty=l1_weights_penalty,
            l1_trades_penalty=l1_trades_penalty,
            linear_constraints=linear_constraints_,
            min_trade_size=min_trade_size,
            max_trades=max_trades,
            max_total_trade_size=max_total_trade_size,
            max_positions=max_positions,
            asset_alphas_uncertainty_cov=_get(asset_alphas_uncertainty_cov, i),
            asset_alphas_uncertainty_kappa=_get(asset_alphas_uncertainty_kappa, i),
            trades_fee=_get(trades_fee, i),
            soft_max_total_trades_fee=_get(soft_max_total_trades_fee, i),
            soft_max_total_trades_fee_penalty=soft_max_total_trades_fee_penalty,
            solver=solver,
            **kwargs,
        )

        res.append(opt_weights)
        if not np.all(np.isnan(opt_weights)):
            is_valid = True

    return np.vstack(res)


def _compute_portfolio_var(weights, loadings: np.ndarray, factor_cov: np.ndarray,
                           specific_var: np.ndarray):
    # use psd_wrap to avoid ARPACK random failure:
    # https://github.com/cvxpy/cvxpy/issues/1424#issuecomment-865967780
    # must convert specific_var to 2D matrix below!
    factor_var = cp.quad_form(loadings @ weights, psd_wrap(factor_cov))
    specific_var = cp.sum_squares(np.sqrt(np.diag(specific_var)) @ weights)
    return factor_var + specific_var


def _numpy_single_period_markowitz_opt(
        asset_cov: Union[np.ndarray, Tuple[
            np.ndarray, np.ndarray, np.ndarray]],
        objective: MarkowitzObj = MarkowitzObj.MAX_RETURN,
        *,
        asset_alphas: Optional[np.ndarray] = None,
        target_vol: Optional[float] = None,
        target_weights: Optional[np.ndarray] = None,
        target_return: Optional[float] = None,
        init_weights: Optional[np.ndarray] = None,
        fixed_weights: Optional[np.ndarray] = None,
        max_leverage: Optional[float] = None,
        min_leverage: Optional[float] = None,
        max_pos_size: Optional[Union[float, np.ndarray]] = None,
        min_pos_size: Optional[float] = 1e-4,
        max_long_pos_size: Optional[Union[float, np.ndarray]] = None,
        max_short_pos_size: Optional[Union[float, np.ndarray]] = None,
        max_trade_size: Optional[Union[float, np.ndarray]] = None,
        soft_max_trade_size: Optional[Union[float, np.ndarray]] = None,
        soft_max_trade_size_penalty: Optional[float] = None,
        buy_veto_mask: Optional[np.ndarray] = None,
        sell_veto_mask: Optional[np.ndarray] = None,
        long_only: bool = False,
        allow_pos_flip: bool = True,
        l1_weights_penalty: Optional[float] = None,
        l1_trades_penalty: Optional[float] = None,
        linear_constraints: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        min_trade_size: Optional[float] = None,
        max_trades: Optional[int] = None,
        max_total_trade_size: Optional[float] = None,
        max_positions: Optional[int] = None,
        asset_alphas_uncertainty_cov: Optional[np.ndarray] = None,
        asset_alphas_uncertainty_kappa: Optional[float] = None,
        trades_fee: Optional[Union[float, np.ndarray]] = None,
        soft_max_total_trades_fee: Optional[float] = None,
        soft_max_total_trades_fee_penalty: Optional[float] = None,
        solver: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
) -> np.ndarray:
    """
    Computes Markowitz optimal portfolio weights.

    Parameters
    ----------
    asset_cov
        Either a full asset cov matrix with shape (N, N), or a tuple of form
        (loadings, factor_cov, specific_var). Loadings must have shape (K, N).
        Factor covariance must have shape (K, K) and specific var must have
        shape (N,).
    asset_alphas
        Forecast asset returns. If forecast return for an asset is NaN,
        its weight is constrained to be 0.
    objective
        Optimization objective.
    target_vol
        Target portfolio volatility (not annualized). Can only be used with
        MAX_RETURN objective.
    target_weights
        Portfolio to minimize tracking error to. Can only be used with
        MIN_TRACKING_ERROR objective.
    target_return
        Target return of portfolio. Can only be used with MIN_VAR objective.
    init_weights
        Initial portfolio positions.
    fixed_weights
        Array of fixed weights to hold. Nan indicates no fixed weight.
    max_leverage
        Maximum gross leverage of portfolio.
    min_leverage
        Minimum gross leverage of portfolio.
    max_pos_size
        Maximum position size. Can be an array to set different max pos size
        for each asset.
    max_long_pos_size
        Maximum long position size. Can be an array to set different value for each
        asset.
    max_short_pos_size
        Maximum short position size. Can be an array to set different value for each
        asset.
    max_trade_size
        Maximum trade size. Can be an array to set different max trade size for
        each asset.
    buy_veto_mask
        Boolean array indicating whether to veto buying each asset. If buying an asset
        is vetoed, then buying is only allowed in order to reduce the asset's pos size
        if it exceeds a pos size limit.
    sell_veto_mask
        Boolean array indicating whether to veto selling each asset. If selling an
        asset is vetoed, then selling is only allowed in order to reduce the asset's
        pos size if it exceeds a pos size limit.
    long_only
        Whether all weights must be positive.
    allow_pos_flip
        Controls whether weight is allowed to have opposite sign to
        expected return.
    l1_weights_penalty
        L1 weights penalty. A higher value will result in sparser portfolios,
        i.e., portfolios with fewer active positions.
    l1_trades_penalty
        L1 trades penalty. A higher value will reduce trading portfolios,
        i.e., portfolios with fewer active positions.
    linear_constraints
        First array in tuple should have shape K x N, where K is the number of
        different constraints and N is the number of assets. Second array should
        contain the lower and upper bounds associated with each constraint, and
        should have shape K x 2.
    min_trade_size
        Minimum trade size as a proportion of FUM. Optimization will take considerably
        longer if set since requires mixed-integer solver.
    max_trades
        Maximum number of trades at each timestep.
    max_total_trade_size
        Maximum total trade size as a proportion of FUM.
    max_positions
        Maximum number of positions at each timestep.
    asset_alphas_uncertainty_cov
        Asset alphas quadratic uncertainty matrix.
    asset_alphas_uncertainty_kappa
        Asset alphas quadratic uncertainty kappa. Maximum allowed value is
        np.sqrt(np.sum((asset_alphas / asset_vol)**2)). If a larger value is
        specified then all weights will be 0.
    trades_fee
        Fee as percent of notional trade value. Can be used to model commission,
        bid/ask spread cost, etc.
    solver
        Solver to use. If not set, CVXPY will choose whichever is most appropriate.
    kwargs
        Additional kwargs to pass to solver.

    Returns
    -------
    Optimal weights.

    Notes
    -----
    If a full asset covariance matrix is provided, assets with nan variance are
    considered invalid. If a structural risk model is provided, assets with nan
    specific variance, any nan loadings, or non-zero exposure to any factor with nan
    variance are considered invalid. Moreover, assets with nan forecast returns/
    target weights are also considered invalid. Invalid assets are excluded from the
    optimization procedure, and the optimal weights returned for them is nan.

    If any of max_trades, max_positions and min_trade_size are set, a mixed-integer
    solver (such as SCIP) must be used. If none of these constraints are specified
    it's reccommended to use the ECOS solver since it's considerably faster.

    If using the SCIP solver, it's recommended to pass scip_params={
    "limits/gap": x, "limits/time": 30} as an additional kwarg, where
    0.01 <= x <= 0.05. Without this, the solver will take a long time.
    """
    # TODO: rename min_trade_size to connote that it is included in the optimization
    #  rather than crudely being applied post-opt (like min_pos_size)
    if asset_alphas is None and objective == MarkowitzObj.MAX_RETURN:
        raise ValueError("Asset returns must be passed for MAX_RETURN objective")
    if target_weights is None and objective == MarkowitzObj.MIN_TRACKING_ERROR:
        raise ValueError("Target weights must be passed for MIN_TRACKING_ERROR"
                         " objective")
    if max_pos_size is not None and np.any(max_pos_size < 0):
        raise ValueError("Max pos sizes must be non-negative")
    if max_long_pos_size is not None and np.any(max_long_pos_size < 0):
        raise ValueError("Max long pos sizes must be non-negative")
    if max_short_pos_size is not None and np.any(max_short_pos_size < 0):
        raise ValueError("Max short pos sizes must be non-negative")

    # use ECOS rather than CLARABEL as default since ECOS seems to be more stable
    if solver is None:
        solver = "ECOS"

    if isinstance(asset_cov, np.ndarray):
        N = asset_cov.shape[0]
        loadings = np.eye(N)
        factor_cov = asset_cov.copy()
        specific_var = np.zeros(N)
    else:
        N = asset_cov[2].shape[0]
        loadings, factor_cov, specific_var = asset_cov

    if isinstance(max_pos_size, float):
        max_pos_size = np.full(N, max_pos_size)
    if isinstance(max_long_pos_size, float):
        max_long_pos_size = np.full(N, max_long_pos_size)
    if isinstance(max_short_pos_size, float):
        max_short_pos_size = np.full(N, max_short_pos_size)
    if isinstance(max_trade_size, float):
        max_trade_size = np.full(N, max_trade_size)
    if isinstance(soft_max_trade_size, float):
        soft_max_trade_size = np.full(N, soft_max_trade_size)
    if isinstance(trades_fee, float):
        trades_fee = np.full(N, trades_fee)

    opt_weights = np.full(N, np.nan)

    # mustn't mark assets with max_pos_size of 0 as invalid - otherwise we'll ignore
    # impact of closing out respective positions when evaluating trade constraints
    invalid_factors = np.isnan(np.diag(factor_cov))
    invalid_assets = ~_numpy_get_valid_assets(loadings, factor_cov, specific_var)
    if asset_alphas is not None:
        invalid_assets = invalid_assets | np.isnan(asset_alphas)
    if target_weights is not None:
        invalid_assets = invalid_assets | np.isnan(target_weights)
    if init_weights is not None:
        invalid_assets = invalid_assets | np.isnan(init_weights)
    if np.all(invalid_assets):
        return opt_weights

    # slice inputs according to invalid mask
    # TODO: perhaps cleaner to set nans to 0 in factor_cov instead of removing
    #  invalid factors from factor cov; then we wouldn't need to compute invalid
    #  factors above
    loadings = loadings[np.ix_(~invalid_factors, ~invalid_assets)]
    factor_cov = factor_cov[np.ix_(~invalid_factors, ~invalid_factors)]
    specific_var = specific_var[~invalid_assets]
    if asset_alphas is not None:
        asset_alphas = asset_alphas[~invalid_assets]
    if target_weights is not None:
        target_weights = target_weights[~invalid_assets]
    if max_pos_size is not None:
        max_pos_size = max_pos_size[~invalid_assets]
    if max_long_pos_size is not None:
        max_long_pos_size = max_long_pos_size[~invalid_assets]
    if max_short_pos_size is not None:
        max_short_pos_size = max_short_pos_size[~invalid_assets]
    if max_trade_size is not None:
        max_trade_size = max_trade_size[~invalid_assets]
    if soft_max_trade_size is not None:
        soft_max_trade_size = soft_max_trade_size[~invalid_assets]
    if buy_veto_mask is not None:
        buy_veto_mask = buy_veto_mask[~invalid_assets]
    if sell_veto_mask is not None:
        sell_veto_mask = sell_veto_mask[~invalid_assets]
    if init_weights is not None:
        init_weights = init_weights[~invalid_assets]
    if fixed_weights is not None:
        fixed_weights = fixed_weights[~invalid_assets]
    if asset_alphas_uncertainty_cov is not None:
        asset_alphas_uncertainty_cov = asset_alphas_uncertainty_cov[
            np.ix_(~invalid_assets, ~invalid_assets)]
    if trades_fee is not None:
        trades_fee = trades_fee[~invalid_assets]

    if objective == MarkowitzObj.MAX_RETURN and np.all(asset_alphas == 0.0):
        return opt_weights

    # must return zero weights explicitly if target vol zero, otherwise solver
    # may return small non-zero weights due to precision issues
    if (objective in [MarkowitzObj.MAX_RETURN, MarkowitzObj.MIN_TRACKING_ERROR] and
            target_vol == 0):
        opt_weights[~invalid_assets] = 0.0
        return opt_weights

    # N_ is number of valid assets
    N_ = loadings.shape[1]

    constraints = []
    opt_weights_var = cp.Variable(N_, name="opt_weights")
    portfolio_var = _compute_portfolio_var(opt_weights_var, loadings, factor_cov,
                                           specific_var)

    if fixed_weights is not None:
        fixed_weights_mask = ~np.isnan(fixed_weights)
        if np.any(fixed_weights_mask):
            constraints.append(opt_weights_var[fixed_weights_mask] == fixed_weights[
                fixed_weights_mask])
    if max_leverage is not None:
        constraints.append(cp.norm(opt_weights_var, 1) <= max_leverage)

    # TODO: allow min leverage constraint for long/short portfolios
    if long_only and min_leverage is not None:
        constraints.append(cp.sum(opt_weights_var) >= min_leverage)

    if max_pos_size is not None:
        # don't apply max pos size constraint where nan
        max_pos_size_mask = ~np.isnan(max_pos_size)
        if np.any(max_pos_size_mask):
            constraints.append(cp.abs(opt_weights_var[max_pos_size_mask])
                               <= max_pos_size[max_pos_size_mask])
    if max_long_pos_size is not None:
        max_long_pos_size_mask = ~np.isnan(max_long_pos_size)
        if np.any(max_long_pos_size_mask):
            constraints.append(opt_weights_var[max_long_pos_size_mask]
                               <= max_long_pos_size[max_long_pos_size_mask])
    if max_short_pos_size is not None:
        max_short_pos_size_mask = ~np.isnan(max_short_pos_size)
        if np.any(max_short_pos_size_mask):
            constraints.append(opt_weights_var[max_short_pos_size_mask]
                               >= -max_short_pos_size[max_short_pos_size_mask])

    if asset_alphas is not None and not allow_pos_flip:
        constraints.append(cp.multiply(opt_weights_var, asset_alphas) >= 0.0)

    if long_only:
        constraints.append(opt_weights_var >= 0.0)

    l1_weights_cost = 0
    if l1_weights_penalty is not None:
        l1_weights_cost = l1_weights_penalty * cp.norm(opt_weights_var, 1)

    if linear_constraints is not None:
        # TODO: convert "greater than" constraints to "less than" ones and only
        #  append single constraint
        constraint_mat = linear_constraints[0][:, ~invalid_assets]
        for i, op in enumerate([operator.ge, operator.le]):
            bounds = linear_constraints[1][:, i]
            bounds_mask = np.isnan(bounds)
            if (~bounds_mask).sum() > 0:
                constraints.append(
                    op(constraint_mat[~bounds_mask] @ opt_weights_var,
                       bounds[~bounds_mask]))

    if max_positions is not None:
        opt_weights_bin = cp.Variable(N_, boolean=True, name="opt_weights_bin")
        constraints += [
            opt_weights_var <= BIG_M * opt_weights_bin,
            opt_weights_var >= -BIG_M * opt_weights_bin,
            cp.sum(opt_weights_bin) <= max_positions,
        ]

    l1_trades_cost = 0
    total_trades_fee_cost = 0
    soft_max_total_trades_fee_cost = 0
    soft_max_trade_size_cost = 0
    if init_weights is not None:
        opt_weights_var.value = init_weights
        opt_trades = opt_weights_var - init_weights

        if l1_trades_penalty is not None:
            l1_trades_cost = l1_trades_penalty * cp.norm(opt_trades, 1)

        if trades_fee is not None:
            total_trades_fee_cost = cp.sum(cp.multiply(cp.abs(opt_trades), trades_fee))
            if (soft_max_total_trades_fee is not None and
                    soft_max_total_trades_fee_penalty is not None):
                soft_max_total_trades_fee_slack = cp.Variable(1)
                soft_max_total_trades_fee_cost = (
                        soft_max_total_trades_fee_slack *
                        soft_max_total_trades_fee_penalty
                )
                constraints += [
                    soft_max_total_trades_fee_slack >= 0,
                    total_trades_fee_cost -
                    soft_max_total_trades_fee_slack <= soft_max_total_trades_fee,
                ]

        if max_total_trade_size is not None:
            constraints.append(cp.norm(opt_trades, 1) <= max_total_trade_size)

        if soft_max_trade_size is not None:
            soft_max_trade_size_mask = ~np.isnan(soft_max_trade_size)
            if np.any(soft_max_trade_size_mask):
                soft_max_trade_size = soft_max_trade_size[soft_max_trade_size_mask]
                soft_max_trade_size_slack = cp.Variable(len(soft_max_trade_size))
                soft_max_trade_size_cost = (
                        cp.norm(soft_max_trade_size_slack, 1) *
                        soft_max_trade_size_penalty)
                constraints += [
                    soft_max_trade_size_slack >= 0,
                    cp.abs(opt_trades[soft_max_trade_size_mask]) -
                    soft_max_trade_size_slack <= soft_max_trade_size
                ]

        if buy_veto_mask is not None:
            if np.any(buy_veto_mask):
                # TODO: account for max_short_pos_size mask too
                buy_veto_max_trade_size = np.maximum(-max_pos_size - init_weights, 0)
                constraints.append(opt_trades[buy_veto_mask] <=
                                   buy_veto_max_trade_size[buy_veto_mask])

        if sell_veto_mask is not None:
            if np.any(sell_veto_mask):
                # TODO: account for max_long_pos_size mask too
                sell_veto_max_trade_size = np.minimum(max_pos_size - init_weights, 0)
                constraints.append(opt_trades[sell_veto_mask] >=
                                   sell_veto_max_trade_size[sell_veto_mask])

        if max_trades is not None:
            opt_trades_bin = cp.Variable(N_, boolean=True, name="opt_trades_bin")
            constraints += [
                opt_trades <= BIG_M * opt_trades_bin,
                opt_trades >= -BIG_M * opt_trades_bin,
                cp.sum(opt_trades_bin) <= max_trades,
            ]

        if max_trade_size is not None:
            # don't apply max trade size constraint where nan
            max_trade_size_mask = ~np.isnan(max_trade_size)
            if np.any(max_trade_size_mask):
                constraints.append(cp.abs(opt_trades[max_trade_size_mask])
                                   <= max_trade_size[max_trade_size_mask])

        if min_trade_size is not None:
            opt_trades_pos = cp.Variable(N_, name="opt_trades_pos")
            opt_trades_neg = cp.Variable(N_, name="opt_trades_neg")
            opt_trades_pos_bin = cp.Variable(N_, boolean=True,
                                             name="opt_trades_pos_bin")
            opt_trades_neg_bin = cp.Variable(N_, boolean=True,
                                             name="opt_trades_neg_bin")

            constraints += [
                opt_trades_pos <= BIG_M * opt_trades_pos_bin,
                opt_trades_pos >= cp.multiply(min_trade_size, opt_trades_pos_bin),
                opt_trades_neg <= BIG_M * opt_trades_neg_bin,
                opt_trades_neg >= cp.multiply(min_trade_size, opt_trades_neg_bin),
                opt_trades_pos_bin + opt_trades_neg_bin <= 1,
                opt_trades == opt_trades_pos - opt_trades_neg,
            ]

    costs = (l1_weights_cost + l1_trades_cost + soft_max_total_trades_fee_cost +
             soft_max_trade_size_cost)
    if objective == MarkowitzObj.MAX_RETURN:
        costs += total_trades_fee_cost
        objective_fn = asset_alphas @ opt_weights_var - costs
        if (asset_alphas_uncertainty_cov is not None and
                asset_alphas_uncertainty_kappa is not None):
            asset_alphas_uncertainty_cov_chol = np.linalg.cholesky(
                asset_alphas_uncertainty_cov)
            objective_fn -= asset_alphas_uncertainty_kappa * cp.norm(
                asset_alphas_uncertainty_cov_chol @ opt_weights_var, 2)
        objective_fn = cp.Maximize(objective_fn)
        if target_vol is not None:
            constraints.append(portfolio_var <= target_vol ** 2)
    elif objective == MarkowitzObj.MIN_VARIANCE:
        objective_fn = cp.Minimize(portfolio_var + costs)
        if asset_alphas is not None and target_return is not None:
            exp_return = asset_alphas @ opt_weights_var
            constraints.append(exp_return >= target_return)
    elif objective == MarkowitzObj.MIN_TRACKING_ERROR:
        active_weights = target_weights - opt_weights_var
        active_var = _compute_portfolio_var(
            active_weights, loadings, factor_cov, specific_var)
        objective_fn = cp.Minimize(active_var + costs)
        if target_vol is not None:
            constraints.append(portfolio_var <= target_vol ** 2)
    else:
        raise ValueError(
            "Objective must either be: MAX_RETURN, MIN_VARIANCE or "
            "MIN_TRACKING_ERROR")

    problem = cp.Problem(objective_fn, constraints)
    problem.solve(solver=solver, verbose=verbose, **kwargs)

    opt_weights_ = opt_weights_var.value
    if opt_weights_ is not None and np.all(~np.isnan(opt_weights_)):
        # TODO: incorporate min pos size into optimization like min trade size?
        if min_pos_size is not None:
            opt_weights_[np.abs(opt_weights_) < min_pos_size] = 0
        time = problem.solver_stats.solve_time
        opt_weights[~invalid_assets] = opt_weights_
        logger.debug(f"Computed Markowitz optimal weights in: {time}s")
        return opt_weights

    raise ValueError("Error computing Markowitz optimal weights")


def _pandas_single_period_markowitz_opt(
        asset_cov: Union[pd.DataFrame, Tuple[
            pd.DataFrame, pd.DataFrame, pd.Series]],
        objective: MarkowitzObj = MarkowitzObj.MAX_RETURN,
        *,
        asset_alphas: Optional[pd.Series] = None,
        target_vol: Optional[float] = None,
        target_weights: Optional[pd.Series] = None,
        target_return: Optional[float] = None,
        init_weights: Optional[pd.Series] = None,
        fixed_weights: Optional[pd.Series] = None,
        max_leverage: Optional[float] = None,
        min_leverage: Optional[float] = None,
        max_pos_size: Optional[Union[float, pd.Series]] = None,
        min_pos_size: Optional[float] = 1e-4,
        max_long_pos_size: Optional[Union[float, pd.Series]] = None,
        max_short_pos_size: Optional[Union[float, pd.Series]] = None,
        max_trade_size: Optional[Union[float, pd.Series]] = None,
        soft_max_trade_size: Optional[Union[float, pd.Series]] = None,
        soft_max_trade_size_penalty: Optional[float] = None,
        buy_veto_mask: Optional[pd.Series] = None,
        sell_veto_mask: Optional[pd.Series] = None,
        long_only: bool = False,
        allow_pos_flip: bool = True,
        l1_weights_penalty: Optional[float] = None,
        l1_trades_penalty: Optional[float] = None,
        linear_constraints: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
        min_trade_size: Optional[float] = None,
        max_trades: Optional[int] = None,
        max_total_trade_size: Optional[float] = None,
        max_positions: Optional[int] = None,
        asset_alphas_uncertainty_cov: Optional[pd.DataFrame] = None,
        asset_alphas_uncertainty_kappa: Optional[float] = None,
        trades_fee: Optional[Union[float, pd.Series]] = None,
        soft_max_total_trades_fee: Optional[float] = None,
        soft_max_total_trades_fee_penalty: Optional[float] = None,
        solver: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
):
    """
    Computes optimal portfolio weights given an objective and constraints.
    """
    if isinstance(asset_cov, pd.DataFrame):
        assets = asset_cov.columns
    else:
        assets = asset_cov[2].index

    opt_weights = _numpy_single_period_markowitz_opt(
        asset_cov=pandas_to_numpy(asset_cov),
        objective=objective,
        asset_alphas=pandas_to_numpy(asset_alphas),
        target_vol=pandas_to_numpy(target_vol),
        target_weights=pandas_to_numpy(target_weights),
        target_return=pandas_to_numpy(target_return),
        init_weights=pandas_to_numpy(init_weights),
        fixed_weights=pandas_to_numpy(fixed_weights),
        max_leverage=max_leverage,
        min_leverage=min_leverage,
        max_pos_size=pandas_to_numpy(max_pos_size),
        min_pos_size=pandas_to_numpy(min_pos_size),
        max_long_pos_size=pandas_to_numpy(max_long_pos_size),
        max_short_pos_size=pandas_to_numpy(max_short_pos_size),
        max_trade_size=pandas_to_numpy(max_trade_size),
        soft_max_trade_size=pandas_to_numpy(soft_max_trade_size),
        soft_max_trade_size_penalty=soft_max_trade_size_penalty,
        buy_veto_mask=pandas_to_numpy(buy_veto_mask),
        sell_veto_mask=pandas_to_numpy(sell_veto_mask),
        long_only=long_only,
        allow_pos_flip=allow_pos_flip,
        l1_weights_penalty=l1_weights_penalty,
        l1_trades_penalty=l1_trades_penalty,
        linear_constraints=pandas_to_numpy(linear_constraints),
        min_trade_size=min_trade_size,
        max_trades=max_trades,
        max_total_trade_size=max_total_trade_size,
        max_positions=max_positions,
        asset_alphas_uncertainty_cov=pandas_to_numpy(asset_alphas_uncertainty_cov),
        asset_alphas_uncertainty_kappa=asset_alphas_uncertainty_kappa,
        trades_fee=trades_fee,
        soft_max_total_trades_fee=soft_max_total_trades_fee,
        soft_max_total_trades_fee_penalty=soft_max_total_trades_fee_penalty,
        solver=solver,
        verbose=verbose,
        **kwargs,
    )

    return pd.Series(opt_weights, index=assets, name="weights")


def single_period_markowitz_opt(
        asset_cov,
        objective=MarkowitzObj.MAX_RETURN,
        *,
        asset_alphas=None,
        target_vol=None,
        target_weights=None,
        target_return=None,
        init_weights=None,
        fixed_weights=None,
        max_leverage=None,
        min_leverage=None,
        max_pos_size=None,
        min_pos_size=1e-4,
        max_long_pos_size=None,
        max_short_pos_size=None,
        max_trade_size=None,
        soft_max_trade_size=None,
        soft_max_trade_size_penalty=None,
        buy_veto_mask=None,
        sell_veto_mask=None,
        long_only=False,
        allow_pos_flip=True,
        l1_weights_penalty=None,
        l1_trades_penalty=None,
        linear_constraints=None,
        min_trade_size: Optional[float] = None,
        max_trades: Optional[int] = None,
        max_total_trade_size: Optional[float] = None,
        max_positions: Optional[int] = None,
        asset_alphas_uncertainty_cov: Optional[pd.DataFrame] = None,
        asset_alphas_uncertainty_kappa: Optional[float] = None,
        trades_fee=None,
        soft_max_total_trades_fee=None,
        soft_max_total_trades_fee_penalty=None,
        solver: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
):
    kwargs_ = {
        "asset_cov": asset_cov,
        "objective": objective,
        "asset_alphas": asset_alphas,
        "target_vol": target_vol,
        "target_weights": target_weights,
        "target_return": target_return,
        "init_weights": init_weights,
        "fixed_weights": fixed_weights,
        "max_leverage": max_leverage,
        "min_leverage": min_leverage,
        "max_pos_size": max_pos_size,
        "soft_max_trade_size": soft_max_trade_size,
        "soft_max_trade_size_penalty": soft_max_trade_size_penalty,
        "min_pos_size": min_pos_size,
        "max_long_pos_size": max_long_pos_size,
        "max_short_pos_size": max_short_pos_size,
        "max_trade_size": max_trade_size,
        "buy_veto_mask": buy_veto_mask,
        "sell_veto_mask": sell_veto_mask,
        "long_only": long_only,
        "allow_pos_flip": allow_pos_flip,
        "l1_weights_penalty": l1_weights_penalty,
        "l1_trades_penalty": l1_trades_penalty,
        "linear_constraints": linear_constraints,
        "min_trade_size": min_trade_size,
        "max_trades": max_trades,
        "max_total_trade_size": max_total_trade_size,
        "max_positions": max_positions,
        "asset_alphas_uncertainty_cov": asset_alphas_uncertainty_cov,
        "asset_alphas_uncertainty_kappa": asset_alphas_uncertainty_kappa,
        "trades_fee": trades_fee,
        "soft_max_total_trades_fee": soft_max_total_trades_fee,
        "soft_max_total_trades_fee_penalty": soft_max_total_trades_fee_penalty,
        "verbose": verbose,
        "solver": solver
    }
    if isinstance(asset_cov, pd.DataFrame) or is_iterable_of(
            asset_cov, (pd.DataFrame, pd.Series)):
        return _pandas_single_period_markowitz_opt(**kwargs_, **kwargs)
    elif isinstance(asset_cov, np.ndarray) or is_iterable_of(asset_cov, np.ndarray):
        return _numpy_single_period_markowitz_opt(**kwargs_, **kwargs)
    raise ValueError("asset_cov must be a array-like or a tuple of array-like objects")
