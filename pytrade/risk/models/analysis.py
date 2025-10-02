import itertools
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Union, Callable

import pandas as pd
from dieboldmariano.dieboldmariano import dm_test
from scipy import stats

from pytrade.portfolio import markowitz_opt
from pytrade.portfolio.analysis import compute_portfolio_returns
from pytrade.portfolio.opt import MarkowitzObj
from pytrade.risk.models.cov import compute_asset_vol, compute_full_asset_cov
from pytrade.risk.models.factor.returns import fit_factor_return_model
from pytrade.risk.models.full import compute_ew_sample_cov
from pytrade.signal.analysis import compute_xs_corr
from pytrade.utils.pandas import stack, unstack

logger = logging.getLogger(__name__)


@dataclass
class FactorAnalytics:
    pvalues: pd.DataFrame
    loadings_corr: pd.DataFrame
    corr2: pd.Series
    factor_returns: pd.DataFrame
    specific_returns: pd.DataFrame
    sample_size: pd.Series
    mv_portfolio_weights: pd.DataFrame
    mv_portfolio_returns: pd.Series

    pvalue_cdf_probs: pd.Series
    corr2_cdf_probs: pd.Series
    loadings_corr_quantiles: pd.Series

    factor_cov: pd.DataFrame
    specific_var: pd.DataFrame
    asset_vol: pd.DataFrame
    dm_test: Optional[pd.DataFrame]


def analyse_factor(
        loadings: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        returns: pd.DataFrame,
        weights: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        *,
        cov_proxy: Optional[pd.DataFrame] = None,
        pvalue_cdf_values: Tuple[float] = (0.05, 0.1, 0.2),
        corr2_cdf_values: Tuple[float] = (0.25, 0.5, 0.75),
        loadings_corr_quantile_levels: Tuple[float] = (0.25, 0.5, 0.75),
        min_nonzero_loadings: int = 10,
        cov_lambda: float = 0.94,
        cov_min_periods: int = 90,
        max_abs_return: Optional[float] = None,
        dm_loss_fn: Optional[Callable[[float, float], float]] = None,
) -> FactorAnalytics:
    """
    Analyses a factor.

    Parameters
    ----------
    loadings
        Must have multiindex of time and factor. Columns must be assets.
    returns
        Returns for each asset in universe.
    weights
        Weights to use for WLS.
    cov_proxy
        Proxy of actual covariance. E.g., realized covariance.
    pvalue_cdf_values
    corr2_cdf_values
    loadings_corr_quantile_levels
    min_nonzero_loadings
    cov_lambda
    cov_min_periods
    max_abs_return
    dm_loss_fn
        Optional loss function to use for DM test. Uses MSE by default. To use
        QLIKE loss function, pass `lambda a, p: a / p - np.log(a / p) - 1`.

    Returns
    -------
    FactorAnalytics
    """
    single = False
    if isinstance(loadings, pd.DataFrame):
        loadings_nlevels = loadings.index.nlevels
        if loadings_nlevels == 2:
            single = True
            loadings = {"model": loadings}
        if loadings_nlevels == 3:
            loadings = unstack(loadings, level="model")

    models = set(loadings.keys())

    if isinstance(weights, pd.DataFrame):
        weights_nlevels = weights.index.nlevels
        if weights_nlevels == 1:
            weights = {k: weights for k in models}
        if weights_nlevels == 2:
            weights = unstack(weights, level="model")

    cov_alpha = 1 - cov_lambda
    adj_returns = returns
    if max_abs_return is not None:
        adj_returns = returns.clip(-max_abs_return, max_abs_return)

    if dm_loss_fn is None:
        dm_loss_fn = lambda a, p: 1.0 / 2 * (a ** 2 - p ** 2) - p * (a - p)

    pvalues = {}
    corr2 = {}
    factor_returns = {}
    specific_returns = {}
    mv_portfolio_weights = {}
    mv_portfolio_returns = {}
    pvalue_cdf_probs = {}
    corr2_cdf_probs = {}
    loadings_corr = {}
    loadings_corr_quantiles = {}
    sample_size = {}
    factor_cov = {}
    specific_var = {}
    asset_vol = {}
    for k in models:
        logger.info(f"Fitting factor model for: {k}")
        model = fit_factor_return_model(adj_returns, loadings[k], weights[k],
                                        min_nonzero_loadings=min_nonzero_loadings)

        loadings_corr[k] = compute_xs_corr(loadings[k])
        loadings_corr_quantiles[k] = loadings_corr[k].stack().groupby(
            level=[1, 2]).quantile(loadings_corr_quantile_levels)

        factor_cov[k] = compute_ew_sample_cov(
            model.factor_returns, alpha=cov_alpha,
            min_periods=cov_min_periods)
        specific_var[k] = model.specific_returns.ewm(
            alpha=cov_alpha, min_periods=cov_min_periods).var()

        # below we compute global minimum variance portfolio
        logger.info(f"Computing minimum variable portfolio for: {k}")
        mv_portfolio_weights[k] = markowitz_opt(
            (loadings[k], factor_cov[k], specific_var[k]),
            objective=MarkowitzObj.MIN_VARIANCE,
            asset_returns=returns,
            min_leverage=1,
            long_only=True)
        mv_portfolio_returns[k] = compute_portfolio_returns(
            mv_portfolio_weights[k], returns)

        asset_vol[k] = compute_asset_vol(loadings[k], factor_cov[k], specific_var[k])
        pvalue_cdf_probs[k] = model.pvalues.apply(lambda x: pd.Series(
            stats.percentileofscore(x, score=pvalue_cdf_values,
                                    nan_policy="omit") / 100.0,
            index=pvalue_cdf_values), axis=0)
        corr2_cdf_probs[k] = pd.Series(
            stats.percentileofscore(
                model.corr2, score=corr2_cdf_values,
                nan_policy="omit") / 100.0, index=corr2_cdf_values)
        pvalues[k] = model.pvalues
        corr2[k] = model.corr2
        sample_size[k] = model.sample_size
        factor_returns[k] = model.factor_returns
        specific_returns[k] = model.specific_returns

    dm_test_ = None
    if len(models) > 1:
        full_asset_cov = {}
        for k in models:
            # must shift asset cov below since var at time T-1 is the prediction of
            # the variance of the return at time T
            # realized var at time T corresponds to variance of return at time T
            full_asset_cov[k] = compute_full_asset_cov(
                loadings[k], factor_cov[k], specific_var[k]).groupby(
                level=1).shift()
        full_asset_cov = stack(full_asset_cov)
        full_asset_cov.index.names = ["time", "asset_1", "model"]
        full_asset_cov.columns.names = ["asset_2"]
        full_asset_cov = (
            full_asset_cov.stack()
            .reorder_levels(["time", "model", "asset_1", "asset_2"])
            .sort_index()
        )
        dm_test_ = {}
        for model_1, model_2 in itertools.combinations(models, 2):
            cov = pd.concat(
                [
                    full_asset_cov.xs(model_1, level="model"),
                    full_asset_cov.xs(model_2, level="model"),
                    cov_proxy,
                ],
                axis=1,
                keys=["model_1", "model_2", "actual"],
            )
            # TODO: shift predicted cov forward?
            cov = cov.dropna(how="any")
            dm_test_[(model_1, model_2)] = cov.groupby(["asset_1", "asset_2"]).apply(
                lambda x: pd.Series(
                    dm_test(x["actual"], x["model_1"], x["model_2"], loss=dm_loss_fn),
                    index=["t_value", "p_value"],
                )
            )
        dm_test_ = stack(dm_test_, names=["model_1", "model_2"])

    def out_(x: Dict, o: int = 1):
        res = x["model"] if single else stack(x, names=["model"])
        if res.index.nlevels > 1:
            return res.swaplevel(-1, o)
        return res

    return FactorAnalytics(
        pvalues=out_(pvalues),
        corr2=out_(corr2),
        factor_returns=out_(factor_returns),
        specific_returns=out_(specific_returns),
        mv_portfolio_weights=out_(mv_portfolio_weights),
        mv_portfolio_returns=out_(mv_portfolio_returns),
        pvalue_cdf_probs=out_(pvalue_cdf_probs, 0),
        corr2_cdf_probs=out_(corr2_cdf_probs, 0),
        loadings_corr=out_(loadings_corr),
        loadings_corr_quantiles=out_(loadings_corr_quantiles),
        sample_size=out_(sample_size),
        factor_cov=out_(factor_cov),
        specific_var=out_(specific_var),
        asset_vol=out_(asset_vol),
        dm_test=dm_test_,
    )
