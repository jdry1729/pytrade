import itertools
import logging
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Union, Callable

import pandas as pd
from dieboldmariano.dieboldmariano import dm_test
from scipy import stats

from pytrade.portfolio import markowitz_opt, analyse_portfolio
from pytrade.portfolio.analysis import compute_portfolio_returns, PortfolioAnalytics
from pytrade.portfolio.opt import MarkowitzObj
from pytrade.risk.models.cov import compute_asset_vol, compute_full_asset_cov, \
    compute_portfolio_cov
import numpy as np
from pytrade.risk.models.factor.returns import fit_factor_return_model, \
    FactorReturnModel, BarraModel
from pytrade.risk.models.full import compute_ew_sample_cov
from pytrade.risk.utils import scale_vol_by_time
from pytrade.signal.analysis import compute_xs_corr
from pytrade.utils.pandas import stack, unstack

logger = logging.getLogger(__name__)


@dataclass
class BarraModelAnalytics:
    pvalues: pd.DataFrame
    vifs: pd.DataFrame
    loadings_corr: pd.DataFrame
    corr2: pd.Series
    factor_returns: pd.DataFrame
    specific_returns: pd.DataFrame
    sample_size: pd.Series
    min_var_portfolio_weights: pd.DataFrame
    min_var_portfolio_returns: pd.Series

    pvalue_cdf_probs: pd.Series
    vif_cdf_probs: pd.Series
    corr2_cdf_probs: pd.Series
    loadings_corr_quantiles: pd.Series
    diebold_test: Optional[pd.DataFrame]
    alpha_portfolio_weights: Optional[pd.DataFrame]
    alpha_portfolio_analytics: Optional[PortfolioAnalytics]


def compute_diebold_test(
        models: Dict[str, BarraModel],
        portfolio_weights: pd.DataFrame,
        portfolio_cov: pd.DataFrame,
        loss_fn: Optional[Callable[[float, float], float]] = None
) -> pd.DataFrame:
    portfolio_cov = portfolio_cov.stack()
    portfolio_cov.index.names = ["time", "portfolio_1", "portfolio_2"]

    model_portfolio_cov = {}
    for k in models:
        # must shift asset cov below since var at time T-1 is the prediction of
        # the variance of the return at time T
        # realized var at time T corresponds to variance of return at time T
        model = models[k]
        model_portfolio_cov[k] = compute_portfolio_cov(
            portfolio_weights, (
                model.loadings, model.factor_cov, model.specific_var)
        ).groupby(level=1).shift()
    model_portfolio_cov = stack(model_portfolio_cov)
    model_portfolio_cov.index.names = ["time", "portfolio_1", "model"]
    model_portfolio_cov.columns.names = ["portfolio_2"]
    model_portfolio_cov = (
        model_portfolio_cov.stack()
        .reorder_levels(["time", "model", "portfolio_1", "portfolio_2"])
        .sort_index()
    )
    res = {}
    for model_1, model_2 in itertools.combinations(models, 2):
        cov = pd.concat(
            [
                model_portfolio_cov.xs(model_1, level="model"),
                model_portfolio_cov.xs(model_2, level="model"),
                portfolio_cov,
            ],
            axis=1,
            keys=["model_1", "model_2", "actual"],
        )
        cov = cov.dropna(how="any")
        res[(model_1, model_2)] = cov.groupby(
            ["portfolio_1", "portfolio_2"]).apply(
            lambda x: pd.Series(
                # negative t-value indicates model 1 prediction is better
                dm_test(x["actual"], x["model_1"], x["model_2"], loss=loss_fn),
                index=["t_value", "p_value"],
            )
        )
    res = stack(res, names=["model_1", "model_2"])
    return res.reorder_levels(("model_1", "model_2", "portfolio_1", "portfolio_2"))


def analyse_barra_model(
        model: Union[BarraModel, Dict[str, BarraModel]],
        asset_prices: pd.DataFrame,
        *,
        pvalue_cdf_values: Tuple[float] = (0.05, 0.1, 0.2),
        vif_cdf_values: Tuple[float] = (1, 2, 5, 10),
        corr2_cdf_values: Tuple[float] = (0.25, 0.5, 0.75),
        loadings_corr_quantile_levels: Tuple[float] = (0.25, 0.5, 0.75),
        diebold_portfolio_weights: Optional[pd.DataFrame] = None,
        diebold_portfolio_cov: Optional[pd.DataFrame] = None,
        diebold_loss_fn: Optional[Callable[[float, float], float]] = None,
        asset_alphas: Optional[pd.DataFrame] = None,
        ann_factor: int = 252,
        target_vol: Optional = None,
) -> BarraModelAnalytics:
    """
    Analyses a Barra model.

    Parameters
    ----------
    model
    asset_prices
    pvalue_cdf_values
    vif_cdf_values
        The rule of thumb is that if VIF is less than 5 then multicollinearity
        isn't a problem.
    corr2_cdf_values
    loadings_corr_quantile_levels
    diebold_portfolio_weights
    diebold_portfolio_cov
        Realized portfolio covariance.
    diebold_loss_fn
        Optional loss function to use for DM test. Uses MSE by default. To use
        QLIKE loss function, pass `lambda a, p: a / p - np.log(a / p) - 1`.
    asset_alphas
        Alphas.
    ann_factor
    target_vol

    Returns
    -------
    BarraModelAnalytics
    """
    logger.info("Analysing Barra models")
    single = False
    models = model
    if isinstance(models, FactorReturnModel):
        models = {"model": models}

    if target_vol is None:
        target_vol = scale_vol_by_time(0.3, 1 / ann_factor)

    asset_returns = asset_prices.pct_change(fill_method=None)

    pvalues = {}
    vifs = {}
    corr2 = {}
    factor_returns = {}
    specific_returns = {}
    pvalue_cdf_probs = {}
    vif_cdf_probs = {}
    corr2_cdf_probs = {}
    loadings_corr = {}
    loadings_corr_quantiles = {}
    sample_size = {}
    min_var_portfolio_weights = {}
    min_var_portfolio_returns = {}
    asset_cov = {}
    for k in models:
        model = models[k]
        asset_cov[k] = (model.loadings, model.factor_cov, model.specific_var)

        loadings_corr[k] = compute_xs_corr(model.loadings)
        loadings_corr_quantiles[k] = loadings_corr[k].stack().groupby(
            level=[1, 2]).quantile(loadings_corr_quantile_levels)

        pvalue_cdf_probs[k] = model.pvalues.apply(lambda x: pd.Series(
            stats.percentileofscore(x, score=pvalue_cdf_values,
                                    nan_policy="omit") / 100.0,
            index=pvalue_cdf_values), axis=0)
        vif_cdf_probs[k] = model.vifs.apply(lambda x: pd.Series(
            stats.percentileofscore(x, score=vif_cdf_values,
                                    nan_policy="omit") / 100.0,
            index=vif_cdf_values), axis=0)
        corr2_cdf_probs[k] = pd.Series(
            stats.percentileofscore(
                model.corr2, score=corr2_cdf_values,
                nan_policy="omit") / 100.0, index=corr2_cdf_values)
        pvalues[k] = model.pvalues
        vifs[k] = model.vifs
        corr2[k] = model.corr2
        sample_size[k] = model.sample_size
        factor_returns[k] = model.factor_returns
        specific_returns[k] = model.specific_returns

    logger.info("Computing minimum variance portfolios")
    for k in models:
        min_var_portfolio_weights[k] = markowitz_opt(
            asset_cov[k],
            objective=MarkowitzObj.MIN_VARIANCE,
            asset_returns=asset_returns,
            min_leverage=1,
            long_only=True)
        # TODO: nan weights prior to first valid index
        min_var_portfolio_returns[k] = compute_portfolio_returns(
            min_var_portfolio_weights[k], asset_returns)

    diebold_test = None
    if (len(models) > 1 and diebold_portfolio_weights is not None
            and diebold_portfolio_cov is not None):
        logger.info("Performing Diebold-Mariano tests")
        diebold_test = compute_diebold_test(
            models,
            portfolio_weights=diebold_portfolio_weights,
            portfolio_cov=diebold_portfolio_cov,
            loss_fn=diebold_loss_fn,
        )

    alpha_portfolio_weights = None
    alpha_portfolio_analytics = None
    if asset_alphas is not None:
        logger.info("Computing characteristic alpha portfolios")
        alpha_portfolio_weights = {}
        for k1 in models:
            # TODO: allow more than 2 levels?
            for k2 in asset_alphas.index.unique(level=1):
                alpha_portfolio_weights[(k1, k2)] = markowitz_opt(
                    asset_cov[k1],
                    asset_alphas=asset_alphas.xs(k2, level=1),
                    target_vol=target_vol,
                )
        alpha_portfolio_weights = stack(alpha_portfolio_weights,
                                        names=["model", "alpha"])
        # TODO: if allow more than two levels I'll need to fix level kwarg below
        first_valid_index = alpha_portfolio_weights.unstack(
            level=(1,2)).dropna(how="any").index[0]
        # must nan weights prior to first valid index so analytics can be compared
        # across models
        alpha_portfolio_weights.loc[
            alpha_portfolio_weights.index.get_level_values(
                "time") < first_valid_index] = np.nan
        alpha_portfolio_analytics = analyse_portfolio(
            alpha_portfolio_weights, asset_prices, ann_factor=ann_factor,
        )

    def out_(x: Dict, o: int = 1):
        if not x:
            return None
        res = x["model"] if single else stack(x, names=["model"])
        if res.index.nlevels > 1:
            return res.swaplevel(-1, o)
        return res

    return BarraModelAnalytics(
        pvalues=out_(pvalues),
        vifs=out_(vifs),
        corr2=out_(corr2),
        factor_returns=out_(factor_returns),
        specific_returns=out_(specific_returns),
        min_var_portfolio_weights=out_(min_var_portfolio_weights),
        min_var_portfolio_returns=out_(min_var_portfolio_returns),
        pvalue_cdf_probs=out_(pvalue_cdf_probs, 0),
        vif_cdf_probs=out_(vif_cdf_probs, 0),
        corr2_cdf_probs=out_(corr2_cdf_probs, 0),
        loadings_corr=out_(loadings_corr),
        loadings_corr_quantiles=out_(loadings_corr_quantiles),
        sample_size=out_(sample_size),
        diebold_test=diebold_test,
        alpha_portfolio_weights=alpha_portfolio_weights,
        alpha_portfolio_analytics=alpha_portfolio_analytics
    )


def fit_ew_risk_model(loadings, factor_returns: pd.DataFrame,
                      specific_returns: pd.DataFrame,
                      cov_lambda: float = 0.94, min_periods: int = 90):
    alpha = 1 - cov_lambda
    factor_cov = compute_ew_sample_cov(factor_returns, alpha=alpha,
                                       min_periods=min_periods)
    specific_var = specific_returns.ewm(alpha=alpha, min_periods=min_periods).var()
    return loadings, factor_cov, specific_var
