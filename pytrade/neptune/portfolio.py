import logging
from typing import Dict, Callable, Any
from typing import Optional
from typing import Union, Tuple

import pandas as pd
from pytrade.neptune import PRICES, FUM, ASSET_DECIMALS
from pytrade.neptune.data import process_signal, reindex
from pytrade.neptune.graph import (TIMES, RETURNS, LIVE_TIME,
                           ASSETS)
from pytrade.neptune.utils import loc
from pytrade.exec.utils import positions_to_trades
from pytrade.graph import add_node, add_alias, NodeRef, node_exists, set_ns, add_edge
from pytrade.portfolio.construction import allocate, vol_scale, weights_to_positions
from pytrade.portfolio.opt import markowitz_opt, MarkowitzObj

logger = logging.getLogger(__name__)


def reindex_allocations(allocations: Union[Dict[str, float], pd.Series, pd.DataFrame],
                        times: pd.DatetimeIndex):
    if isinstance(allocations, Dict):
        return pd.DataFrame(allocations, index=times)
    if isinstance(allocations, pd.Series):
        return pd.DataFrame(allocations.to_dict(), index=times)
    return allocations.reindex(index=times)


def signal_to_alpha(signal: pd.DataFrame, asset_vol: pd.DataFrame):
    return signal * asset_vol


def combine_portfolios(portfolio_weights: Dict[str, NodeRef],
                       allocations: Union[Dict[str, float], NodeRef]):
    allocations = add_node(reindex_allocations, "allocations",
                           args=(allocations, TIMES))
    # contributions are computed for visibility
    contributions = add_node(allocate, "contributions",
                             args=(portfolio_weights, allocations),
                             combine=False)
    nodes = set()
    if node_exists(LIVE_TIME):
        with set_ns("live"):
            nodes.add(add_node(loc, "allocs", args=(allocations, LIVE_TIME)))
            nodes.add(add_node(
                lambda x, t: x.xs(t, level=0), "contributions",
                args=(contributions, LIVE_TIME)))

    weights = add_node(lambda x: x.groupby(level=0).sum(),
                       "weights", args=(contributions,))
    # ensure live allocs/ contributions computed when
    for node in nodes:
        add_edge(node, weights)
    return weights


def add_portfolio(fn: Union[NodeRef, Callable[..., NodeRef]],
                  asset_cov: Optional[
                      Union[NodeRef, Tuple[NodeRef, NodeRef, NodeRef]]] = None,
                  objective: Optional[MarkowitzObj] = None,
                  *,
                  target_vol: Optional[float] = None,
                  reindex_limit: Optional[int] = 0,
                  ffill_limit: Optional[int] = 0,
                  max_leverage: Optional[float] = None,
                  init_weights: Optional[NodeRef] = None,
                  fixed_weights: Optional[NodeRef] = None,
                  max_pos_size: Optional[Union[float, NodeRef]] = None,
                  max_long_pos_size: Optional[Union[float, NodeRef]] = None,
                  max_short_pos_size: Optional[Union[float, NodeRef]] = None,
                  max_trade_size: Optional[Union[float, NodeRef]] = None,
                  soft_max_trade_size: Optional[Union[float, NodeRef]] = None,
                  soft_max_trade_size_penalty: Optional[float] = None,
                  l1_weights_penalty: Optional[float] = None,
                  l1_trades_penalty: Optional[float] = None,
                  linear_constraints: Optional[NodeRef] = None,
                  min_trade_size: Optional[float] = None,
                  max_trades: Optional[int] = None,
                  max_total_trade_size: Optional[float] = None,
                  max_positions: Optional[int] = None,
                  asset_alphas_uncertainty_cov: Optional[NodeRef] = None,
                  asset_alphas_uncertainty_kappa: Optional[
                      Union[float, NodeRef]] = None,
                  trades_fee: Optional[NodeRef] = None,
                  soft_max_total_trades_fee: Optional[NodeRef] = None,
                  soft_max_total_trades_fee_penalty: Optional[float] = None,
                  solver: Optional[str] = None,
                  solver_params: Optional[Dict] = None,
                  analytics_fn: Optional[
                      Callable[[NodeRef], Dict[str, NodeRef]]] = None,
                  args: Tuple[Any] = (),
                  **kwargs) -> NodeRef:
    if solver_params is None:
        solver_params = {}

    target_weights = None
    asset_alphas = None
    opt_weights = None

    reindex_kwargs = dict(index=TIMES, columns=ASSETS, method="ffill",
                          limit=reindex_limit)
    process_kwargs = dict(ffill_limit=ffill_limit)
    if objective == MarkowitzObj.MAX_RETURN:
        with set_ns("alpha"):
            with set_ns("raw"):
                with set_ns("unaligned"):
                    asset_alphas = fn(*args, **kwargs) if callable(fn) else fn
                asset_alphas = add_node(reindex, "values", args=(asset_alphas,),
                                        **reindex_kwargs)
            asset_alphas = add_node(
                process_signal, "values", args=(asset_alphas,), **process_kwargs)
    elif objective is None or objective == MarkowitzObj.MIN_TRACKING_ERROR:
        with set_ns("target"):
            with set_ns("raw"):
                with set_ns("unaligned"):
                    target_weights = fn(*args, **kwargs) if callable(fn) else fn
                target_weights = add_node(reindex, "weights", args=(target_weights,),
                                          **reindex_kwargs)
            target_weights = add_node(
                process_signal, "weights", args=(target_weights,), **process_kwargs)

    with set_ns("opt"):
        if objective is None:
            if target_vol is not None:
                opt_weights = add_node(
                    vol_scale, "weights", args=(target_weights, asset_cov),
                    target_vol=target_vol
                )
        else:
            opt_weights = add_node(
                markowitz_opt,
                "weights",
                asset_cov=asset_cov,
                objective=objective,
                asset_alphas=asset_alphas,
                asset_returns=RETURNS,
                target_vol=target_vol,
                target_weights=target_weights,
                init_weights=init_weights,
                fixed_weights=fixed_weights,
                max_leverage=max_leverage,
                max_pos_size=max_pos_size,
                max_long_pos_size=max_long_pos_size,
                max_short_pos_size=max_short_pos_size,
                max_trade_size=max_trade_size,
                soft_max_trade_size=soft_max_trade_size,
                soft_max_trade_size_penalty=soft_max_trade_size_penalty,
                l1_weights_penalty=l1_weights_penalty,
                l1_trades_penalty=l1_trades_penalty,
                linear_constraints=linear_constraints,
                min_trade_size=min_trade_size,
                max_trades=max_trades,
                max_total_trade_size=max_total_trade_size,
                max_positions=max_positions,
                asset_alphas_uncertainty_cov=asset_alphas_uncertainty_cov,
                asset_alphas_uncertainty_kappa=asset_alphas_uncertainty_kappa,
                trades_fee=trades_fee,
                soft_max_total_trades_fee=soft_max_total_trades_fee,
                soft_max_total_trades_fee_penalty=soft_max_total_trades_fee_penalty,
                solver=solver,
                **solver_params
            )

    weights = opt_weights
    if weights is None:
        weights = target_weights

    weights = add_alias(weights, "weights")
    positions = add_node(weights_to_positions, "positions",
                         args=(weights, PRICES, FUM, ASSET_DECIMALS))
    add_node(positions_to_trades, "trades", args=(positions,))

    # TODO: raise if analytic tries to add node at higher level
    if analytics_fn is not None:
        with set_ns("analytics"):
            with set_ns("raw"):
                analytics = analytics_fn(weights)
            for k in analytics:
                if isinstance(analytics[k], NodeRef):
                    add_alias(analytics[k], k)

    with set_ns("live"):
        add_node(loc, "weights", args=(weights, LIVE_TIME), series_name="weights")

    return weights
