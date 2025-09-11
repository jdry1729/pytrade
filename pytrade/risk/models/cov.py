from typing import Union, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from pytrade.utils.collections import is_iterable_of
from pytrade.utils.pandas import pandas_to_numpy, stack

# TODO: what's this for?
np.seterr(divide="ignore", invalid="ignore")


def _numpy_get_valid_assets(loadings: np.ndarray, factor_cov: np.ndarray,
                            specific_var: np.ndarray) -> np.ndarray:
    """
    Returns a mask indicating whether each asset is valid with respect to the
    specified risk model (i.e., its volatility can be computed).

    Parameters
    ----------
    loadings
        Loadings.
    factor_cov
        Factor covariance matrix.
    specific_var
        Specific variance.

    Returns
    -------
    Valid mask.
    """
    invalid_assets = np.any(np.isnan(loadings), axis=-2)
    invalid_assets = invalid_assets | np.isnan(specific_var)
    invalid_factors = np.isnan(np.diagonal(factor_cov, axis1=-1, axis2=-2))
    return ~(invalid_assets | np.any(
        (np.abs(loadings) > 0) & np.expand_dims(invalid_factors, -1), axis=-2))


def _numpy_compute_portfolio_cov(portfolio_weights: np.ndarray,
                                 asset_cov: Union[np.ndarray, Tuple[
                                     np.ndarray, np.ndarray, np.ndarray]]
                                 ) -> np.ndarray:
    """
    Computes portfolio covariance.

    Parameters
    ----------
    portfolio_weights
        Portfolio weights. Must have shape (T, J, N), where T is number of time
        steps, J is number of portfolios and N is number of assets.
    asset_cov
        Either a full asset cov matrix with shape (T, N, N), or a tuple of form
        (loadings, factor_cov, specific_var). Loadings must have shape (T, K, N).
        Factor covariance must have shape (T, K, K) and specific var must have
        shape (T, N).

    Returns
    -------
    out : (T, J, J)
        Portfolio covariance.

    Notes
    -----
    If a full asset covariance matrix is provided, assets with nan variance are
    considered invalid. If a structural risk model is provided, assets with nan
    specific variance or any nan loadings are considered invalid. If a portfolio
    has a non-zero exposure to an invalid asset, the covariances for that portfolio
    will all be nan in the result.
    """
    if isinstance(asset_cov, np.ndarray):
        loadings = None
        factor_cov = asset_cov
        specific_var = None
        invalid_assets_mask = np.isnan(np.diagonal(factor_cov, axis1=1, axis2=2))
        factor_cov = np.nan_to_num(factor_cov)
    else:
        loadings, factor_cov, specific_var = asset_cov
        invalid_assets_mask = (np.isnan(specific_var) |
                               np.any(np.isnan(loadings), axis=1))
        loadings = np.nan_to_num(loadings)
        factor_cov = np.nan_to_num(factor_cov)
        specific_var = np.nan_to_num(specific_var)

    portfolio_weights = np.nan_to_num(portfolio_weights)
    weights_mask = np.abs(portfolio_weights) > 0
    invalid_cov_mask = np.any(
        weights_mask & np.expand_dims(invalid_assets_mask, 1), axis=2)
    invalid_cov_mask = np.logical_or(
        np.expand_dims(invalid_cov_mask, axis=2),
        np.expand_dims(invalid_cov_mask, axis=1))

    x1 = portfolio_weights
    if loadings is not None:
        x1 = np.einsum("tij,tkj->tik", portfolio_weights, loadings)

    x2 = np.einsum("tip,tpk->tik", x1, factor_cov)
    x3 = np.einsum("tik,tjk->tij", x2, x1)

    x5 = 0
    if specific_var is not None:
        x4 = np.einsum("tin,tn->tin", portfolio_weights, specific_var)
        x5 = np.einsum("tin,tjn->tij", x4, portfolio_weights)

    return np.where(invalid_cov_mask, np.nan, x3 + x5)


def _pandas_compute_portfolio_cov(
        portfolio_weights: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        asset_cov: Union[pd.DataFrame, Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    if isinstance(portfolio_weights, dict):
        portfolio_weights = stack(portfolio_weights, names=["portfolio"])
    portfolio_cov = _numpy_compute_portfolio_cov(
        pandas_to_numpy(portfolio_weights), pandas_to_numpy(asset_cov))
    return pd.DataFrame(np.row_stack(portfolio_cov),
                        index=portfolio_weights.index,
                        columns=portfolio_weights.index.unique(1))


def compute_portfolio_cov(portfolio_weights, asset_cov):
    # TODO: also check asset cov?
    if isinstance(portfolio_weights, (pd.DataFrame, dict)):
        return _pandas_compute_portfolio_cov(portfolio_weights, asset_cov)
    elif isinstance(portfolio_weights, np.ndarray):
        return _numpy_compute_portfolio_cov(portfolio_weights, asset_cov)
    raise ValueError(
        "portfolio_weights and asset_cov must either both be dataframes or"
        " both be numpy arrays")


def _numpy_compute_portfolio_vol(
        portfolio_weights: np.ndarray,
        asset_cov: Union[
            np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> pd.DataFrame:
    """
    Computes portfolio vol.

    Parameters
    ----------
    portfolio_weights
        Portfolio weights. May be of shape (T, N) or (T, J, N) where J is the
        number of portfolios, and N is number of assets.
    asset_cov
        Either a full asset cov matrix with shape (T, N, N), or a tuple of form
        (loadings, factor_cov, specific_var). Loadings must have shape (T, K, N).
        Factor covariance must have shape (T, K, K) and specific var must have
        shape (T, N).

    Returns
    -------
    out: (T, ...)
        Portfolio vol. If portfolio weights is two-dimensional, will have
        shape (T,). Otherwise will have shape (T, J).
    """
    squeeze = False
    if portfolio_weights.ndim == 2:
        squeeze = True
        portfolio_weights = np.expand_dims(portfolio_weights, axis=1)

    portfolio_cov = _numpy_compute_portfolio_cov(portfolio_weights, asset_cov)
    portfolio_vol = np.sqrt(np.diagonal(portfolio_cov, axis1=1, axis2=2))

    if squeeze:
        return portfolio_vol.squeeze()
    return portfolio_vol


def _pandas_compute_portfolio_vol(
        portfolio_weights: pd.DataFrame, asset_cov: Union[pd.DataFrame, Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """
    Computes portfolio vol.

    Returns
    -------
    Portfolio vol. If portfolio weights index has 2 levels, then portfolio
    vol will also have 2 levels.
    """
    portfolio_vol = _numpy_compute_portfolio_vol(
        pandas_to_numpy(portfolio_weights), pandas_to_numpy(asset_cov))
    # flatten needed below since series data must be 1-dimensional
    return pd.Series(np.row_stack(portfolio_vol).flatten(),
                     index=portfolio_weights.index)


def compute_portfolio_vol(portfolio_weights, asset_cov):
    # TODO: check asset_cov type too
    array_like = [portfolio_weights]
    if is_iterable_of(array_like, pd.DataFrame):
        return _pandas_compute_portfolio_vol(portfolio_weights, asset_cov)
    elif is_iterable_of(array_like, np.ndarray):
        return _numpy_compute_portfolio_vol(portfolio_weights, asset_cov)
    raise ValueError(
        "portfolio_weights and asset_cov must either both be dataframes or"
        " both be numpy arrays")


def _numpy_compute_single_period_full_asset_cov(
        loadings: np.ndarray,
        factor_cov: np.ndarray,
        specific_var: np.ndarray) -> np.ndarray:
    return loadings.T @ factor_cov @ loadings + np.diag(specific_var)


def _numpy_compute_full_asset_cov(loadings: np.ndarray, factor_cov: np.ndarray,
                                  specific_var: np.ndarray) -> np.ndarray:
    """
    Computes full asset covariance matrix from loadings, factor covariance and
    specific var.

    Parameters
    ----------
    loadings
        Loadings. Must have shape (T, K, N).
    factor_cov
        Must have shape (T, K, K), where T is number of time steps and K is number
         of factors.
    specific_var
        Specific var. Must have shape (T, N).

    Returns
    -------
    out: (T x N x N)
        Full asset covariance matrix.
    """
    valid_assets = _numpy_get_valid_assets(loadings, factor_cov, specific_var)
    valid_mask = np.expand_dims(valid_assets, -1) & np.expand_dims(valid_assets, 1)

    loadings = np.nan_to_num(loadings)
    factor_cov = np.nan_to_num(factor_cov)
    specific_var = np.nan_to_num(specific_var)

    N = specific_var.shape[1]
    specific_var = np.expand_dims(specific_var, 1) * np.eye(N)
    x1 = np.einsum("tpn,tpk->tnk", loadings, factor_cov)
    x2 = np.einsum("tnk,tkm->tnm", x1, loadings)
    return np.where(valid_mask, x2 + specific_var, np.nan)


def _pandas_compute_full_asset_cov(loadings: pd.DataFrame,
                                   factor_cov: pd.DataFrame,
                                   specific_var: pd.DataFrame) -> pd.DataFrame:
    times = loadings.index.unique(0)
    assets = specific_var.columns
    asset_cov = _numpy_compute_full_asset_cov(
        pandas_to_numpy(loadings), pandas_to_numpy(factor_cov),
        pandas_to_numpy(specific_var))
    return pd.DataFrame(np.row_stack(asset_cov),
                        index=pd.MultiIndex.from_product([times, assets]),
                        columns=assets)


def compute_full_asset_cov(loadings, factor_cov, specific_var):
    array_like = [loadings, factor_cov, specific_var]
    if is_iterable_of(array_like, pd.DataFrame):
        return _pandas_compute_full_asset_cov(loadings, factor_cov, specific_var)
    elif is_iterable_of(array_like, np.ndarray):
        return _numpy_compute_full_asset_cov(loadings, factor_cov, specific_var)
    raise ValueError(
        "loadings, factor_cov and specific_var must either all be dataframes or all"
        " be numpy arrays")


def _numpy_compute_asset_var(loadings: np.ndarray, factor_cov: np.ndarray,
                             specific_var: np.ndarray) -> np.ndarray:
    """
    Computes asset variance from loadings, factor covariance, and specific var.

    Parameters
    ----------
    loadings : np.ndarray
        Loadings. Must have shape (T, K, N).
    factor_cov : np.ndarray
        Factor covariance matrix. Must have shape (T, K, K), where T is the number
        of time steps and K is the number of factors.
    specific_var : np.ndarray
        Specific variance. Must have shape (T, N).

    Returns
    -------
    out : np.ndarray
        Asset variance with shape (T, N).

    Notes
    -----
    Much more memory-efficient to use this function to compute asset variance vs
    computing full covariance and then taking diagonal.
    """
    valid_assets = _numpy_get_valid_assets(loadings, factor_cov, specific_var)

    loadings = np.nan_to_num(loadings)
    factor_cov = np.nan_to_num(factor_cov)
    specific_var = np.nan_to_num(specific_var)

    factor_contrib = np.einsum("tpn,tpk,tkn->tn", loadings, factor_cov, loadings)
    asset_var = factor_contrib + specific_var
    return np.where(valid_assets, asset_var, np.nan)


def _pandas_compute_asset_var(loadings: pd.DataFrame,
                              factor_cov: pd.DataFrame,
                              specific_var: pd.DataFrame) -> pd.DataFrame:
    asset_var = _numpy_compute_asset_var(
        pandas_to_numpy(loadings), pandas_to_numpy(factor_cov),
        pandas_to_numpy(specific_var))
    return pd.DataFrame(asset_var, index=specific_var.index,
                        columns=specific_var.columns)


def compute_asset_var(loadings, factor_cov, specific_var):
    array_like = [loadings, factor_cov, specific_var]
    if is_iterable_of(array_like, pd.DataFrame):
        return _pandas_compute_asset_var(loadings, factor_cov, specific_var)
    elif is_iterable_of(array_like, np.ndarray):
        return _numpy_compute_asset_var(loadings, factor_cov, specific_var)
    raise ValueError(
        "loadings, factor_cov and specific_var must either all be dataframes or all"
        " be numpy arrays")


def compute_asset_vol(loadings, factor_cov, specific_var):
    return np.sqrt(compute_asset_var(loadings, factor_cov, specific_var))


def _numpy_compute_asset_beta(portfolio_weights: np.ndarray,
                              asset_cov: Union[np.ndarray, Tuple[
                                  np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Computes asset beta to a number of factor portfolios.

    Parameters
    ----------
    portfolio_weights
        Portfolio weights. May have shape (T, N) or (T, J, N).
    asset_cov
        Either a full asset cov matrix with shape (T, N, N), or a tuple of form
        (loadings, factor_cov, specific_var). Loadings must have shape (T, K, N).
        Factor covariance must have shape (T, K, K) and specific var must have
        shape (T, N).

    Returns
    -------
    out: (T, ...)
        Asset beta. If portfolio weights is two-dimensional, will have
        shape (T, N). Otherwise will have shape (T, J, N).
    """
    # TODO: create single get_invalid_assets function which accepts asset_cov as
    #  input and re-use that here, and everywhere else!
    squeeze = False
    if portfolio_weights.ndim == 2:
        squeeze = True
        portfolio_weights = np.expand_dims(portfolio_weights, axis=1)

    if isinstance(asset_cov, np.ndarray):
        loadings = None
        factor_cov = asset_cov
        specific_var = None
        invalid_assets_mask = np.isnan(np.diagonal(factor_cov, axis1=1, axis2=2))
        factor_cov = np.nan_to_num(factor_cov)
    else:
        loadings, factor_cov, specific_var = asset_cov
        invalid_assets_mask = (np.isnan(specific_var) |
                               np.any(np.isnan(loadings), axis=1))
        loadings = np.nan_to_num(loadings)
        factor_cov = np.nan_to_num(factor_cov)
        specific_var = np.nan_to_num(specific_var)

    portfolio_weights = np.nan_to_num(portfolio_weights)
    weights_mask = np.abs(portfolio_weights) > 0

    # we cannot compute beta of asset to a portfolio if asset is invalid or portfolio
    # has non-zero exposure to any invalid asset
    invalid_beta_mask = np.any(
        weights_mask & np.expand_dims(invalid_assets_mask, 1), axis=2)
    invalid_beta_mask = np.expand_dims(invalid_beta_mask, -1) | np.expand_dims(
        invalid_assets_mask, 1)

    x1 = portfolio_weights.transpose(0, 2, 1)
    if loadings is not None:
        x1 = np.einsum("tkn,tjn->tkj", loadings, portfolio_weights)

    x2 = np.einsum("tkp,tpj->tkj", factor_cov, x1)

    x3 = x2
    if loadings is not None:
        x3 = np.einsum("tkn,tkj->tnj", loadings, x2)

    x4 = 0
    if specific_var is not None:
        x4 = np.einsum("tn,tjn->tnj", specific_var, portfolio_weights)

    x5 = x3 + x4

    # TODO: portfolio cov could easily be computed from x5 at this point; I should
    #  use this code in _numpy_compute_portfolio_cov
    portfolio_var = np.einsum("tjn,tnj->tj", portfolio_weights, x5)
    beta_mat = x5 / np.expand_dims(portfolio_var, 1)
    beta_mat = beta_mat.transpose(0, 2, 1)
    beta_mat = np.where(invalid_beta_mask, np.nan, beta_mat)

    if squeeze:
        return beta_mat.squeeze()
    return beta_mat


def _pandas_compute_asset_beta(
        portfolio_weights: pd.DataFrame,
        asset_cov: Union[pd.DataFrame, Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    beta_mat = _numpy_compute_asset_beta(
        pandas_to_numpy(portfolio_weights),
        pandas_to_numpy(asset_cov))
    return pd.DataFrame(np.row_stack(beta_mat), index=portfolio_weights.index,
                        columns=portfolio_weights.columns)


def compute_asset_beta(portfolio_weights, asset_cov):
    """
    Computes beta of each asset to a set of portfolios. To compute the beta of
    a portfolio to one of the portfolios for which weights are given, you just have
    to compute the dot product between the portfolio's weights and the betas of
    each asset at each timestep.

    Parameters
    ----------
    portfolio_weights
        Portfolio weights.
    asset_cov
        Asset covariance.

    Returns
    -------
    Asset beta.
    """
    array_like = [portfolio_weights]
    if is_iterable_of(array_like, pd.DataFrame):
        return _pandas_compute_asset_beta(portfolio_weights, asset_cov)
    elif is_iterable_of(array_like, np.ndarray):
        return _numpy_compute_asset_beta(portfolio_weights, asset_cov)
    raise ValueError("portfolio_weights must be a dataframe or numpy arrays")


def _numpy_compute_factor_beta(
        portfolio_weights: np.ndarray,
        factor_portfolio_weights: np.ndarray,
        asset_cov: Union[np.ndarray, Tuple[
            np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Computes beta of a portfolio to a number of factor portfolios.

    Parameters
    ----------
    portfolio_weights
        Must have shape (T, N).
    factor_portfolio_weights
        Must have shape (T, J, N).
    asset_cov
        Either a full asset cov matrix with shape (T, N, N), or a tuple of form
        (loadings, factor_cov, specific_var). Loadings must have shape (T, K, N).
        Factor covariance must have shape (T, K, K) and specific var must have
        shape (T, N).

    Returns
    -------
    Beta of portfolio to factor portfolios.
    """
    asset_beta = _numpy_compute_asset_beta(factor_portfolio_weights, asset_cov)
    if factor_portfolio_weights.ndim == 3:
        portfolio_weights = np.expand_dims(portfolio_weights, 1)
    invalid_beta_mask = np.any((np.abs(portfolio_weights) > 0) &
                               np.isnan(asset_beta), axis=-1)
    invalid_beta_mask |= np.all(np.isnan(portfolio_weights), axis=-1)
    beta = np.nansum(portfolio_weights * asset_beta, axis=-1)
    return np.where(invalid_beta_mask, np.nan, beta)


def _pandas_compute_factor_beta(
        portfolio_weights: pd.DataFrame,
        factor_portfolio_weights: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        asset_cov: Union[pd.DataFrame, Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> pd.Series:
    if isinstance(factor_portfolio_weights, dict):
        factor_portfolio_weights = stack(factor_portfolio_weights, names=["factor"])
    beta = _numpy_compute_factor_beta(pandas_to_numpy(portfolio_weights),
                                      pandas_to_numpy(factor_portfolio_weights),
                                      pandas_to_numpy(asset_cov))
    # return series to match _pandas_compute_portfolio_vol
    return pd.Series(np.row_stack(beta).flatten(),
                     index=factor_portfolio_weights.index)


def compute_factor_beta(portfolio_weights, factor_portfolio_weights, asset_cov):
    # TODO: check factor_portfolio_weights is dataframe or dict of dataframe
    array_like = [portfolio_weights]
    if is_iterable_of(array_like, pd.DataFrame):
        return _pandas_compute_factor_beta(portfolio_weights,
                                           factor_portfolio_weights,
                                           asset_cov)
    elif is_iterable_of(array_like, np.ndarray):
        return _numpy_compute_factor_beta(portfolio_weights,
                                          factor_portfolio_weights,
                                          asset_cov)
    raise ValueError(
        "portfolio_weights and factor_portfolio_weights must either both"
        " be dataframes or both be numpy arrays")


def _numpy_compute_factor_corr(portfolio_weights: np.ndarray,
                               factor_portfolio_weights: np.ndarray,
                               asset_cov: Union[np.ndarray, Tuple[
                                   np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    beta = _numpy_compute_factor_beta(portfolio_weights,
                                      factor_portfolio_weights,
                                      asset_cov)
    factor_vol = _numpy_compute_portfolio_vol(factor_portfolio_weights, asset_cov)
    portfolio_vol = _numpy_compute_portfolio_vol(portfolio_weights, asset_cov)
    return beta * factor_vol / np.expand_dims(portfolio_vol, -1)


def _pandas_compute_factor_corr(
        portfolio_weights: pd.DataFrame,
        factor_portfolio_weights: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        asset_cov: Union[pd.DataFrame, Tuple[
            pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> pd.Series:
    if isinstance(factor_portfolio_weights, dict):
        factor_portfolio_weights = stack(factor_portfolio_weights, names=["factor"])
    corr = _numpy_compute_factor_corr(pandas_to_numpy(portfolio_weights),
                                      pandas_to_numpy(factor_portfolio_weights),
                                      pandas_to_numpy(asset_cov))
    # return series to match _pandas_compute_portfolio_vol
    return pd.Series(np.row_stack(corr).flatten(),
                     index=factor_portfolio_weights.index)


def compute_factor_corr(portfolio_weights, factor_portfolio_weights, asset_cov):
    array_like = [portfolio_weights]
    if is_iterable_of(array_like, pd.DataFrame):
        return _pandas_compute_factor_corr(
            portfolio_weights, factor_portfolio_weights, asset_cov)
    elif is_iterable_of(array_like, np.ndarray):
        return _numpy_compute_factor_corr(
            portfolio_weights, factor_portfolio_weights, asset_cov)
    raise ValueError(
        "portfolio_weights and factor_portfolio_weights must either both"
        " be dataframes or both be numpy arrays")


def compute_tracking_error(portfolio_weights: pd.DataFrame,
                           target_weights: pd.DataFrame,
                           asset_cov: Union[pd.DataFrame, Tuple[
                               pd.DataFrame, pd.DataFrame, pd.DataFrame]],
                           ann_factor: Optional[float] = None) -> pd.Series:
    error = compute_portfolio_vol(portfolio_weights - target_weights, asset_cov)
    if ann_factor is not None:
        error *= np.sqrt(ann_factor)
    return error
