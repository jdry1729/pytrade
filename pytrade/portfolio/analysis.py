import logging
from dataclasses import dataclass
from typing import Optional, Iterable, Dict, Union, List, Tuple, Any, Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from pandas.tseries.frequencies import to_offset
from pytrade.data.arctic import write_data, read_data
from pytrade.data.utils import leading_zeros_to_nan
from pytrade.utils.position import positions_to_trades
from pytrade.portfolio.construction import weights_to_positions, ignore_small_trades, \
    positions_to_weights
from pytrade.stats.tests import ttest_1samp
from pytrade.utils.pandas import stack, flatten_index, to_frame, \
    count_nonzero
from pytrade.utils.plotting import plot
from pytrade.utils.profile import load_profile
from pytrade.utils.random import generate_uid
from scipy.stats import skew, kurtosis, zscore

logger = logging.getLogger(__name__)


def read_portfolio(portfolio_id: str):
    profile = load_profile()
    weights_lib = profile.portfolio_weights_lib
    return read_data(weights_lib, f"{portfolio_id}/weights")


def write_portfolio(portfolio_weights: pd.DataFrame, metadata: Dict[str, Any]):
    profile = load_profile()
    weights_lib = profile.portfolio_weights_lib
    portfolio_id = generate_uid()
    logger.info(
        f"Writing portfolio weights to {weights_lib}; ID={portfolio_id}")
    write_data(weights_lib, f"{portfolio_id}/weights",
               portfolio_weights, create_library=True, metadata=metadata)
    # with sqlalchemy_engine().begin() as conn:
    #     conn.execute(insert(portfolios_table).values(
    #         id=portfolio_id, name=name, universe=universe, freq=freq,
    #         start_time=start_time, end_time=end_time))


@dataclass
class PortfolioAnalytics:
    weights: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    gross_leverage: pd.Series
    net_leverage: pd.Series
    # TODO: rename to weights turnover?
    turnover: pd.Series
    information_horizon: pd.Series
    rolling_volatility: pd.Series
    drawdown: pd.Series
    nonzero_positions: pd.Series
    portfolio_returns: pd.Series
    portfolio_returns_by_asset: pd.Series
    costs: pd.DataFrame
    summary: pd.Series

    @property
    def cum_portfolio_returns(self):
        cum_portfolio_returns = ts_cumsum(self.portfolio_returns)
        cum_portfolio_returns.name = "return"
        return cum_portfolio_returns

    @property
    def cum_portfolio_returns_by_asset(self):
        cum_portfolio_returns = ts_cumsum(self.portfolio_returns_by_asset)
        cum_portfolio_returns.name = "return"
        return cum_portfolio_returns


def compute_gross_leverage(portfolio_weights: pd.DataFrame):
    """
    Computes gross leverage.

    Parameters
    ----------
    portfolio_weights
        May have multiindex or single index.

    Returns
    -------
    Gross leverage.
    """
    nan_mask = portfolio_weights.isnull().all(1)
    leverage = portfolio_weights.abs().sum(axis=1)
    leverage.name = "leverage"
    return leverage.mask(nan_mask)


def compute_net_leverage(portfolio_weights: pd.DataFrame):
    """
    Computes net leverage.

    Parameters
    ----------
    portfolio_weights
        May have multiindex or single index.

    Returns
    -------
    Net leverage.
    """
    nan_mask = portfolio_weights.isnull().all(1)
    leverage = portfolio_weights.sum(axis=1)
    leverage.name = "leverage"
    return leverage.mask(nan_mask)


def compute_weights_turnover(portfolio_weights: pd.DataFrame,
                             ann_factor: Optional[float] = None):
    """
    Computes weights turnover.

    Parameters
    ----------
    portfolio_weights
        Portfolio weights.
    ann_factor
        Annualization factor.

    Notes
    -----

    """
    squeeze = False
    if portfolio_weights.index.nlevels == 1:
        squeeze = True
        portfolio_weights = stack([portfolio_weights])

    # divide by 2 since roughly half of the weight changes will result in
    # positions being closed, which we don't want to count as turnover
    levels = list(range(1, portfolio_weights.index.nlevels))
    turnover = portfolio_weights.fillna(0).groupby(
        level=levels).diff().abs().sum(axis=1)
    # set turnover at first time step to nan
    # TODO: handle case where portfolios start at different times
    first_time = turnover.index.get_level_values(0)[0]
    turnover.loc[first_time] = np.nan

    nan_mask = portfolio_weights.isnull().all(axis=1)
    nan_mask = nan_mask & nan_mask.groupby(level=levels).shift(fill_value=True)
    turnover = turnover.mask(nan_mask) / 2.0
    if ann_factor is not None:
        turnover *= ann_factor

    if squeeze:
        turnover = turnover.xs(0, level=1)

    turnover.name = "turnover"

    return turnover


def compute_sharpe_ratio(portfolio_returns: pd.Series,
                         ann_factor: Optional[float] = None,
                         ignore_leading_zeros: bool = True):
    # TODO: need copy?
    portfolio_returns = portfolio_returns.copy()

    # TODO: create decorator for below?
    squeeze = False
    if portfolio_returns.index.nlevels == 1:
        squeeze = True
        portfolio_returns = stack([portfolio_returns])

    # groupby all levels except first which is time
    levels = list(range(1, portfolio_returns.index.nlevels))

    if ignore_leading_zeros:
        portfolio_returns = portfolio_returns.groupby(
            level=levels, group_keys=False).apply(leading_zeros_to_nan)

    sharpe_ratio = portfolio_returns.groupby(
        level=levels).mean() / portfolio_returns.groupby(level=levels).std()
    # must annualize sharpe ratio for comparision's sake
    if ann_factor is not None:
        sharpe_ratio *= np.sqrt(ann_factor)

    if squeeze:
        sharpe_ratio = sharpe_ratio.loc[0]

    return sharpe_ratio


def compute_skewness(portfolio_returns: pd.Series):
    # TODO: use decorator for below
    squeeze = False
    if portfolio_returns.index.nlevels == 1:
        squeeze = True
        portfolio_returns = stack([portfolio_returns])

    levels = list(range(1, portfolio_returns.index.nlevels))
    skewness = portfolio_returns.groupby(level=levels).apply(
        skew, nan_policy="omit")

    if squeeze:
        skewness = skewness.loc[0]

    return skewness


def compute_rolling_beta(portfolio_returns: pd.Series,
                         factor_portfolio_returns: pd.Series,
                         window: int = 30) -> pd.Series:
    """
    Computes rolling beta.

    Parameters
    ----------
    portfolio_returns
    factor_portfolio_returns
    window

    Returns
    -------
    Rolling beta.

    Notes
    -----
    Annualized beta is the same as 1D beta.
    """
    return (portfolio_returns.rolling(window)
            .cov(factor_portfolio_returns)
            .divide(factor_portfolio_returns.rolling(window).var(), axis=0))


def compute_kurtosis(portfolio_returns: pd.Series):
    # TODO: use decorator for below
    squeeze = False
    if portfolio_returns.index.nlevels == 1:
        squeeze = True
        portfolio_returns = stack([portfolio_returns])

    levels = list(range(1, portfolio_returns.index.nlevels))
    res = portfolio_returns.groupby(level=levels).apply(
        kurtosis, nan_policy="omit")

    if squeeze:
        res = res.loc[0]

    return res


def compute_information_horizon(portfolio_weights: pd.DataFrame,
                                asset_returns: pd.DataFrame, max_lag=10,
                                ann_factor: Optional[float] = None
                                ):
    squeeze = False
    nlevels = portfolio_weights.index.nlevels
    if nlevels == 1:
        squeeze = True
        portfolio_weights = stack([portfolio_weights])

    sharpe_ratios = []
    levels = list(range(1, nlevels))
    for lag in range(0, max_lag):
        portfolio_returns = compute_portfolio_returns(
            portfolio_weights.groupby(level=levels).shift(lag), asset_returns)
        sharpe_ratios.append(
            compute_sharpe_ratio(portfolio_returns, ann_factor))

    information_horizon = stack(sharpe_ratios, names=["lag"], sort_level="lag")
    information_horizon = information_horizon.reorder_levels(
        [-1] + list(range(nlevels - 1)))

    if squeeze:
        information_horizon = information_horizon.xs(0, level=1)

    information_horizon.name = "info_horizon"

    return information_horizon


def compute_rolling_vol(portfolio_returns: pd.Series, alpha: float = 0.03,
                        min_periods: int = 20,
                        ann_factor: Optional[float] = None):
    # TODO: create decorator for below?
    squeeze = False
    if portfolio_returns.index.nlevels == 1:
        squeeze = True
        portfolio_returns = stack([portfolio_returns])

    # use groupby().apply() rather than groupby().rolling() due to bug:
    #  https://github.com/pandas-dev/pandas/issues/14013
    levels = list(range(1, portfolio_returns.index.nlevels))
    # use alpha=0.03 to mimic RiskMetrics model
    rolling_vol = portfolio_returns.groupby(level=levels,
                                            group_keys=False).apply(
        lambda x: x.ewm(alpha=alpha, min_periods=min_periods).std())
    if ann_factor is not None:
        rolling_vol *= np.sqrt(ann_factor)

    if squeeze:
        rolling_vol = rolling_vol.xs(0, level=1)

    return rolling_vol


def compute_vol(portfolio_returns: pd.Series,
                ann_factor: Optional[float] = None):
    # TODO: use decorator for below
    squeeze = False
    if portfolio_returns.index.nlevels == 1:
        squeeze = True
        portfolio_returns = stack([portfolio_returns])

    levels = list(range(1, portfolio_returns.index.nlevels))
    vol = portfolio_returns.groupby(level=levels).std()
    if ann_factor is not None:
        vol *= np.sqrt(ann_factor)

    if squeeze:
        vol = vol.loc[0]

    return vol


def compute_tstat(portfolio_returns: pd.Series):
    squeeze = False
    if portfolio_returns.index.nlevels == 1:
        squeeze = True
        portfolio_returns = stack([portfolio_returns])

    levels = list(range(1, portfolio_returns.index.nlevels))
    tstat = portfolio_returns.groupby(level=levels).apply(ttest_1samp)

    if squeeze:
        tstat = tstat.xs(0, level=0)

    return tstat


def compute_nonzero_positions(portfolio_weights: pd.DataFrame, threshold: float = 1e-5):
    nan_mask = portfolio_weights.isnull().all(1)
    nonzero_positions = portfolio_weights.abs().gt(threshold).sum(1)
    nonzero_positions.name = "nonzero_positions"
    return nonzero_positions.mask(nan_mask)


def compute_drawdown(portfolio_returns: pd.Series):
    # below assumes constant FUM
    # need to add 1 to below so we are looking at drawdown of portfolio value
    # series rather than of pnl series
    # TODO: create decorator for below?
    squeeze = False
    if portfolio_returns.index.nlevels == 1:
        squeeze = True
        portfolio_returns = stack([portfolio_returns])

    levels = list(range(1, portfolio_returns.index.nlevels))
    pnl = (portfolio_returns.groupby(level=levels).cumsum() + 1)
    drawdown = pnl / pnl.groupby(level=levels).cummax() - 1.0

    if squeeze:
        drawdown = drawdown.xs(0, level=1)

    return drawdown


def compute_mean(data: pd.Series, ignore_leading_zeros: bool = True):
    # TODO: use decorator for below
    squeeze = False
    if data.index.nlevels == 1:
        squeeze = True
        data = stack([data])

    levels = list(range(1, data.index.nlevels))

    if ignore_leading_zeros:
        data = data.groupby(level=levels, group_keys=False).apply(
            leading_zeros_to_nan)

    mean = data.groupby(level=levels).mean()

    if squeeze:
        mean = mean.loc[0]

    return mean


def compute_max_drawdown(drawdown: pd.Series):
    # TODO: use decorator for below
    squeeze = False
    if drawdown.index.nlevels == 1:
        squeeze = True
        drawdown = stack([drawdown])

    levels = list(range(1, drawdown.index.nlevels))
    max_drawdown = abs(drawdown.groupby(level=levels).min())

    if squeeze:
        max_drawdown = max_drawdown.loc[0]

    return max_drawdown


def get_funding_fees(asset_prices: pd.DataFrame, funding_rate: pd.DataFrame):
    """
    Get funding fees.

    Parameters
    ----------
    asset_prices
        Asset prices.
    funding_rate
        "Absolute" funding rate, i.e., the amount participants with long positions
        should pay to those with short positions per contract over the period.
        It's assumed the frequency of the funding payments is constant.

    Returns
    -------
    Funding fees.

    Notes
    -----
    Given portfolio weights with the same freq as asset prices, the funding payments
    can be calculated as weights.shift() * fees.
    """
    # better to use to_offset and inferred_freq below rather than just freq because
    # sometimes the freq of the datetime index won't be set
    asset_prices_freq = to_offset(asset_prices.index.inferred_freq)
    funding_rate_freq = to_offset(funding_rate.index.inferred_freq)
    if funding_rate_freq < asset_prices_freq:
        asset_prices = asset_prices.resample(funding_rate_freq).ffill()
    elif funding_rate_freq > asset_prices_freq:
        sf = funding_rate_freq / asset_prices_freq
        funding_rate = funding_rate.resample(asset_prices_freq).ffill()
        funding_rate = funding_rate / sf
    # don't need to multiply by -1 below since returning a fee
    # the funding accrued from T to T + 1 is realized at T + 1
    fees = (funding_rate / asset_prices).shift()
    return fees.resample(asset_prices_freq, closed="right", label="right").sum()


def compute_ib_commission(trades: pd.DataFrame, asset_prices: pd.DataFrame,
                          *, fee_per_unit_trade: float = 0.0035,
                          min_trade_fee: Optional[float] = 0.35,
                          max_trade_fee_pct: Optional[float] = 0.01,
                          by_asset: bool = False) -> pd.Series:
    """
    Computes IB-style commission in accounting currency.

    Parameters
    ----------
    trades
        Trades.
    asset_prices
        Asset prices.
    fee_per_unit_trade
        Absolute fee per share in the accounting currency.
    min_trade_fee
        Absolute minimum fee incurred per trade.
    max_trade_fee_pct
        Maximum fee incurred per trade. Takes precedence over min trade fee.
    by_asset
        Whether to return results by asset.


    Returns
    -------
    Commission.
    """
    # TODO: allow trades and asset_prices to be series
    squeeze = False
    if trades.index.nlevels == 1:
        squeeze = True
        trades = stack([trades])

    abs_trades = trades.abs()
    notional = abs_trades * asset_prices

    commission = abs_trades.multiply(fee_per_unit_trade, axis=1)
    if min_trade_fee is not None:
        commission = commission.where(
            commission == 0, commission.clip(lower=min_trade_fee))
    if max_trade_fee_pct is not None:
        commission = commission.clip(upper=max_trade_fee_pct * notional)

    if not by_asset:
        commission = commission.sum(axis=1).rename("commission")

    if squeeze:
        commission = commission.xs(0, level=1)
    return commission


def compute_trades_fee(
        trades: pd.DataFrame,
        asset_prices: pd.DataFrame,
        fee_pct: pd.DataFrame,
) -> pd.Series:
    """
    Computes trading fee.

    Parameters
    ----------
    trades
        Trades.
    asset_prices
        Asset prices.
    fee_pct
        Fee as percentage of notional.

    Returns
    -------
    Series of fees.
    """
    trades_usd = trades.abs().mul(asset_prices, level=0)
    return trades_usd.mul(fee_pct, level=0).sum(axis=1)


def ib_costs_fn(positions: pd.DataFrame, asset_prices: pd.DataFrame,
                fee_per_unit_trade: float = 0.0035,
                min_trade_fee: Optional[float] = 0.35,
                max_trade_fee_pct: Optional[float] = 0.01,
                bid_ask_spread: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    # TODO: add borrow fees
    trades = positions_to_trades(positions)
    commission = compute_ib_commission(
        trades, asset_prices, fee_per_unit_trade=fee_per_unit_trade,
        min_trade_fee=min_trade_fee, max_trade_fee_pct=max_trade_fee_pct)
    if bid_ask_spread is not None:
        # must divide bid_ask_spread by 2!
        bid_ask_spread_cost = compute_trades_fee(
            trades, asset_prices, bid_ask_spread / 2.0).rename(
            "bid_ask_spread_cost")
        return pd.concat([commission, bid_ask_spread_cost], axis=1)
    return commission


# TODO: make this more general? allow other fns, not just cumsum
def ts_cumsum(s: pd.Series):
    squeeze = False
    if s.index.nlevels == 1:
        squeeze = True
        s = stack([s])

    levels = list(range(1, s.index.nlevels))
    cumsum = s.groupby(level=levels).cumsum()

    if squeeze:
        cumsum = cumsum.xs(0, level=1)

    return cumsum


def compute_large_return_periods(
        returns: pd.Series, window: int = 30, num_periods: Optional[int] = None,
        large_return_threshold: Optional[float] = 0.1
):
    """

    Parameters
    ----------
    returns
    window
    num_periods
    large_return_threshold

    Returns
    -------
    Large return periods.

    Notes
    -----
    This can be applied to a dataframe of multi-indexed series of returns
    using::

       returns.groupby(level=1, group_keys=True).apply(
       compute_large_return_periods, window=30, num_periods=10)
    """
    if num_periods is None and large_return_threshold is None:
        raise ValueError("Error computing large return periods; either num_periods or"
                         " large_return_threshold must be specified")

    # nlevels may be > 1 if passing this function in groupby apply
    nlevels = returns.index.nlevels
    if nlevels > 1:
        returns = returns.droplevel(list(range(1, nlevels)))
    large_returns = []
    rolling_returns = returns.rolling(window).sum()
    abs_rolling_returns = rolling_returns.abs()
    i = 0
    while True:
        end_time = abs_rolling_returns.idxmax()
        # must use pd.isnull rather than np.isnan since latter raises
        # exception for non-float inputs
        if pd.isnull(end_time):
            break
        end_iloc = abs_rolling_returns.index.get_loc(end_time)
        start_time = abs_rolling_returns.index[end_iloc - window + 1]

        period_return = rolling_returns.iloc[end_iloc]
        if large_return_threshold is not None and abs(
                period_return) < large_return_threshold:
            break

        abs_rolling_returns.iloc[
        (end_iloc + 1 - window): (end_iloc + 1 + window)
        ] = np.nan
        large_returns.append(
            {
                "start": start_time,
                "end": end_time,
                "return": period_return,
            }
        )

        i += 1
        if num_periods is not None and i >= num_periods:
            break
    return pd.DataFrame(large_returns)


# TODO: allow positions to be passed here too?
def analyse_portfolio(
        portfolio_weights: pd.DataFrame,
        asset_prices: pd.DataFrame,
        costs_fn: Optional[
            Callable[[pd.DataFrame], Union[pd.DataFrame, pd.Series]]] = None,
        fum: float = 10000,
        position_decimals: Optional[int] = None,
        min_trade_size: Optional[float] = None,
        ann_factor: int = 260,
        info_horizon_max_lag: int = 100
):
    """
    Analyses a portfolio.

    Parameters
    ----------
    portfolio_weights
        Can have arbitrary number of levels.
    asset_prices
        Asset prices.
    costs_fn
        Function to compute trading costs. Must accept positions at each timestep
        as argument, and return a dataframe/ series of costs. If a dataframe is
        returned, it's assumed each column represents a different cost type, and
        the total cost will be computed by summing the columns. The costs must
        be expressed in units of the accounting currency, rather than as a percentage
        of FUM.
    fum
        FUM to use.
    position_decimals
        Decimals to use to compute positions.
    min_trade_size
        Minimum trade size to assume when computing positions.
    ann_factor
        Annualization factor.
    info_horizon_max_lag
        Maximum lag to use when computing the information horizon.

    Returns
    -------
    Portfolio analytics.

    Notes
    -----
    Portfolio weights can have an arbitrary number of levels.
    """
    # TODO: compute return correlation heatmap
    asset_returns = asset_prices.pct_change(fill_method=None)

    positions = weights_to_positions(portfolio_weights, asset_prices, fum)
    if position_decimals is not None or min_trade_size is not None:
        prac_positions = positions
        if position_decimals is not None:
            prac_positions = prac_positions.round(decimals=position_decimals)
        if min_trade_size is not None:
            prac_positions = ignore_small_trades(
                prac_positions, asset_prices, min_trade_size)
        positions = stack([positions, prac_positions], keys=["theo", "prac"],
                          names="style")
        prac_portfolio_weights = positions_to_weights(
            prac_positions, asset_prices, fum)
        portfolio_weights = stack([portfolio_weights, prac_portfolio_weights],
                                  keys=["theo", "prac"], names="style")

    trades = positions_to_trades(positions)

    gross_leverage = compute_gross_leverage(portfolio_weights)
    net_leverage = compute_net_leverage(portfolio_weights)
    nonzero_positions = count_nonzero(portfolio_weights).rename(
        "nonzero_positions")
    weights_turnover = compute_weights_turnover(
        portfolio_weights, ann_factor=ann_factor)

    costs = None
    portfolio_returns = compute_portfolio_returns(portfolio_weights, asset_returns)
    if costs_fn is not None:
        portfolio_returns = {"gross": portfolio_returns}
        costs = costs_fn(positions) / fum
        total_costs = costs.copy()
        if isinstance(total_costs, pd.DataFrame):
            total_costs = total_costs.sum(axis=1)
        # TODO: apply leading zeros to nan to each group in net
        portfolio_returns["net"] = portfolio_returns["gross"].subtract(
            total_costs, fill_value=0)
        portfolio_returns = stack(portfolio_returns, names="costs")
    portfolio_returns.name = "return"

    gross_portfolio_returns_by_asset = compute_portfolio_returns(
        portfolio_weights, asset_returns, by_asset=True)

    historical_drawdown = compute_drawdown(portfolio_returns)
    rolling_vol = compute_rolling_vol(portfolio_returns, ann_factor=ann_factor)
    # for now, we ignore costs when computing information horizon
    info_horizon = compute_information_horizon(
        portfolio_weights, asset_returns, max_lag=info_horizon_max_lag,
        ann_factor=ann_factor)

    # summary
    summary = {
        "max_drawdown": compute_max_drawdown(historical_drawdown),
        "tstat": compute_tstat(portfolio_returns).xs("pvalue", level=-1),
        "volatility": compute_vol(portfolio_returns, ann_factor=ann_factor),
        "sharpe_ratio": compute_sharpe_ratio(
            portfolio_returns, ann_factor=ann_factor),
        "skewness": compute_skewness(portfolio_returns),
        "kurtosis": compute_kurtosis(portfolio_returns),
        "turnover": compute_mean(weights_turnover),
        "gross_leverage": compute_mean(gross_leverage),
        "net_leverage": compute_mean(net_leverage),
    }

    if costs_fn is not None:
        summary["turnover"] = stack(
            [summary["turnover"]] * 2, keys=["gross", "net"], names="costs")
        summary["cost"] = stack([compute_mean(
            total_costs)], keys=["net"], names="costs")
        summary["gross_leverage"] = stack(
            [summary["gross_leverage"]] * 2, keys=["gross", "net"], names="costs")
        summary["net_leverage"] = stack(
            [summary["net_leverage"]] * 2, keys=["gross", "net"], names="costs")
    summary = stack(summary, names="metric", sort_level=None)
    summary.name = "summary"

    return PortfolioAnalytics(
        weights=portfolio_weights,
        positions=positions,
        trades=trades,
        gross_leverage=gross_leverage,
        net_leverage=net_leverage,
        turnover=weights_turnover,
        nonzero_positions=nonzero_positions,
        portfolio_returns=portfolio_returns,
        portfolio_returns_by_asset=gross_portfolio_returns_by_asset,
        drawdown=historical_drawdown,
        rolling_volatility=rolling_vol,
        information_horizon=info_horizon,
        costs=costs,
        summary=summary,
    )


def plot_summary_table(analytics, ax):
    ax.axis("off")
    summary = flatten_index(analytics.summary.unstack(
        level=list(range(0, analytics.summary.index.nlevels - 1))),
        axis=1).applymap(
        "{:,.2f}".format).reset_index(names="")
    ax.set_title("Summary", loc="left")
    tab = ax.table(cellText=summary.values,
                   colLabels=summary.columns,
                   colColours=["lightgrey"] * len(summary.columns),
                   loc="upper left",
                   bbox=[0, 0, 1, 1])
    tab.auto_set_column_width(col=list(range(len(summary.columns) + 1)))


def plot_cum_return(analytics, ax):
    N = analytics.portfolio_returns.index.nlevels
    levels_1 = list(range(1, N))
    locator = mdates.YearLocator()
    # TODO: have unstacked_portfolio_returns attribute on analytics?
    plot(ax,
         flatten_index(
             analytics.portfolio_returns.unstack(level=levels_1).cumsum(),
             axis=1),
         title="Cumulative Return", xaxis_locator=locator)


def plot_drawdown(analytics, ax):
    N = analytics.portfolio_returns.index.nlevels
    levels_1 = list(range(1, N))
    locator = mdates.YearLocator()
    plot(ax,
         flatten_index(analytics.drawdown.unstack(level=levels_1), axis=1),
         title="Drawdown", xaxis_locator=locator)


def plot_rolling_vol(analytics, ax):
    N = analytics.portfolio_returns.index.nlevels
    levels_1 = list(range(1, N))
    locator = mdates.YearLocator()
    plot(ax,
         flatten_index(analytics.rolling_volatility.unstack(level=levels_1),
                       axis=1).filter(regex=".*gross"),
         title="Rolling Volatility (Gross)", xaxis_locator=locator)


def plot_turnover(analytics, ax):
    N = analytics.portfolio_returns.index.nlevels
    levels_2 = list(range(1, N - 1))
    locator = mdates.YearLocator()
    plot(ax,
         flatten_index(to_frame(analytics.turnover.unstack(level=levels_2)),
                       axis=1),
         title="Turnover", xaxis_locator=locator)


# TODO: create general function for plotting gross/ net leverage, nonzero
# positions, etc.

def plot_gross_leverage(analytics, ax):
    N = analytics.portfolio_returns.index.nlevels
    levels_2 = list(range(1, N - 1))
    locator = mdates.YearLocator()
    plot(ax,
         flatten_index(
             to_frame(analytics.gross_leverage.unstack(level=levels_2)),
             axis=1),
         title="Gross Leverage", xaxis_locator=locator)


def plot_net_leverage(analytics, ax):
    N = analytics.portfolio_returns.index.nlevels
    levels_2 = list(range(1, N - 1))
    locator = mdates.YearLocator()
    plot(ax,
         flatten_index(to_frame(analytics.net_leverage.unstack(level=levels_2)),
                       axis=1),
         title="Net Leverage", xaxis_locator=locator)


def plot_nonzero_positions(analytics, ax):
    N = analytics.portfolio_returns.index.nlevels
    levels_2 = list(range(1, N - 1))
    locator = mdates.YearLocator()
    plot(ax,
         flatten_index(
             to_frame(analytics.nonzero_positions.unstack(level=levels_2)),
             axis=1),
         title="Non-zero Positions", xaxis_locator=locator)


def plot_cum_costs(analytics, ax):
    N = analytics.portfolio_returns.index.nlevels
    levels_2 = list(range(1, N - 1))
    locator = mdates.YearLocator()
    plot(ax,
         flatten_index(
             to_frame(
                 analytics.costs.unstack(level=levels_2)).cumsum(),
             axis=1),
         title="Cumulative Costs", xaxis_locator=locator)


def plot_info_horizon(analytics, ax):
    N = analytics.portfolio_returns.index.nlevels
    levels_2 = list(range(1, N - 1))
    information_horizon = to_frame(
        analytics.information_horizon.unstack(level=levels_2))
    ax.set_title("Information Horizon", loc="left")
    ax.grid(True)
    ax.plot(information_horizon)
    ax.legend(information_horizon.columns)


def plot_portfolio_returns(analytics, ax):
    N = analytics.portfolio_returns.index.nlevels
    levels_1 = list(range(1, N))
    locator = mdates.YearLocator()
    plot(ax,
         flatten_index(analytics.portfolio_returns.unstack(level=levels_1),
                       axis=1).filter(
             regex=".*gross"),
         title="Portfolio Returns (Gross)", xaxis_locator=locator)


def plot_returns_dist(analytics, ax):
    N = analytics.portfolio_returns.index.nlevels
    levels_1 = list(range(1, N))
    # TODO: add t-statistics/ p-values in legend
    ax.set_title("Returns Distribution", loc="left")
    # TODO: extract below to function
    dist_returns = flatten_index(
        analytics.portfolio_returns.unstack(level=levels_1), axis=1).filter(
        regex=".*gross")
    returns_robust = dist_returns[
        abs(zscore(dist_returns, nan_policy="omit")) < 3]
    sns.kdeplot(returns_robust, fill=True, ax=ax)
    ax.grid(True)
    # below makes grid appear behind plot
    ax.set_axisbelow(True)
    ax.set_ylabel(None)


TEARSHEET_ITEMS = {
    "summary": plot_summary_table,
    "cum_return": plot_cum_return,
    "drawdown": plot_drawdown,
    "rolling_vol": plot_rolling_vol,
    "returns_dist": plot_returns_dist,
    "turnover": plot_turnover,
    "info_horizon": plot_info_horizon,
    "gross_leverage": plot_gross_leverage,
    "net_leverage": plot_net_leverage,
    "nonzero_positions": plot_nonzero_positions,
    "cum_costs": plot_cum_costs,
    "portfolio_returns": plot_portfolio_returns,
}


def get_grid_shape(layout: Iterable[Union[List, Tuple]]):
    idxs = [[], []]
    for x in layout:
        for i in range(2):
            if isinstance(x[i], slice):
                if x[i].start is not None:
                    idxs[i].append(x[i].start + 1)
                if x[i].stop is not None:
                    idxs[i].append(x[i].stop)
            else:
                idxs[i].append(x[i] + 1)
    return max(idxs[0]), max(idxs[1])


def create_tearsheet(analytics: PortfolioAnalytics,
                     layout: Optional[Dict] = None,
                     height_ratios: Optional[Iterable] = None,
                     figsize: Optional[Iterable] = None):
    if layout is None:
        layout = {
            "summary": (0, slice(None, 3)),
            "cum_return": (1, 0),
            "drawdown": (1, 1),
            "turnover": (1, 2),
            "rolling_vol": (1, 3),
            "gross_leverage": (2, 0),
            "net_leverage": (2, 1),
            "nonzero_positions": (2, 2),
            "info_horizon": (2, 3)
        }
        height_ratios = [1] + [2] * 2

    nrows, ncols = get_grid_shape(layout.values())
    if figsize is None:
        figsize = (0.5 + 5 * ncols, 2.5 * nrows)

    plt.rcParams.update({"font.size": 8, "font.family": "Verdana"})
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=plt.get_cmap("tab10").colors)

    fig = plt.figure(layout="constrained", figsize=figsize)
    gs = GridSpec(nrows=nrows, ncols=ncols, figure=fig, hspace=0.075,
                  height_ratios=height_ratios)

    for k in TEARSHEET_ITEMS:
        if k in layout:
            TEARSHEET_ITEMS[k](analytics, fig.add_subplot(gs[layout[k]]))

    # close figure so it isn't displayed twice in jupyter
    plt.close(fig)
    return fig


def compute_portfolio_returns(portfolio_weights: pd.DataFrame,
                              asset_returns: pd.DataFrame,
                              by_asset: bool = False) -> pd.Series:
    """
    Computes portfolio returns, ignoring fees.

    Parameters
    ----------
    portfolio_weights
        Portfolio weights. Must have multi-index of time and portfolio.
    asset_returns
        Asset returns.
    by_asset
        Whether to compute the returns for each asset.

    Returns
    -------
    Portfolio returns.

    Notes
    -----
    Assumes weights at time T are held for the period T to T+1, and returns
    at time T are for the period from T-1 to T.

    If all weights at a moment in time are nan, the corresponding portfolio
    return will be nan. If even one of the weights is non-nan, the
    portfolio return will not be nan.
    """
    # if portfolio weights isn't multi-indexed, we make it so and squeeze the
    # output
    squeeze = False
    if portfolio_weights.index.nlevels == 1:
        squeeze = True
        portfolio_weights = stack([portfolio_weights])

    # specify level=0 below so multiplication is broadcast across time axis
    levels = list(range(1, portfolio_weights.index.nlevels))
    portfolio_returns = portfolio_weights.groupby(level=levels).shift().mul(
        asset_returns, level=0)

    if not by_asset:
        # skipna=True so return computed even when weights/ returns contain NaNs
        # (e.g., because asset doesn't exist yet)
        nan_mask = portfolio_returns.isnull().all(axis=1)
        portfolio_returns = portfolio_returns.sum(axis=1, skipna=True)
        # set portfolio return to nan if return for all assets is nan!
        portfolio_returns = portfolio_returns.mask(nan_mask)

    if squeeze:
        portfolio_returns = portfolio_returns.xs(0, level=1)

    # TODO: how to handle inf?
    portfolio_returns = portfolio_returns.replace([np.inf, -np.inf], np.nan)

    return portfolio_returns
