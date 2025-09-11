# import numpy as np
# import pandas as pd
#
# from pytrade.risk.structural import _numpy_compute_portfolio_vol
# from pytrade.utils.collections import is_iterable_of
# from pytrade.utils.pandas import pandas_to_numpy
# from pytrade.utils.pandas import stack
#
#
# TODO: fix after changing portfolio cov/ vol functions to use loadings, factor_cov
#  and specific_var
# def _numpy_mctr(portfolio_weights: np.ndarray, asset_cov: np.ndarray):
#     """
#     Computes marginal contribution to risk.
#
#     Parameters
#     ----------
#     portfolio_weights : (T, M, N)
#         Portfolio weights.
#     asset_cov : (T, N, N)
#         Asset covaraince.
#
#     Returns
#     -------
#     out : (T, M, N)
#         Marginal contribution to risk.
#     """
#     hV = np.einsum("tij,tjk->tik", portfolio_weights, asset_cov)
#     portfolio_vol = _numpy_compute_portfolio_vol(portfolio_weights,
#                                                  asset_cov)
#     return np.divide(hV, portfolio_vol)
#
#
# def _pandas_mctr(portfolio_weights: pd.DataFrame, asset_cov: pd.DataFrame):
#     mctr = _numpy_mctr(pandas_to_numpy(portfolio_weights),
#                        pandas_to_numpy(asset_cov))
#     return pd.DataFrame(np.row_stack(mctr), index=portfolio_weights.index,
#                         columns=portfolio_weights.columns)
#
#
# def mctr(portfolio_weights, asset_cov):
#     array_like = [portfolio_weights, asset_cov]
#     if is_iterable_of(array_like, pd.DataFrame):
#         return _pandas_mctr(portfolio_weights, asset_cov)
#     elif is_iterable_of(array_like, np.ndarray):
#         return _numpy_mctr(portfolio_weights, asset_cov)
#     raise ValueError(
#         "portfolio_weights and asset_cov must either both be dataframes"
#         " or all be numpy arrays")
#
#
# def _numpy_mctr_risk_decomp(portfolio_weights: np.ndarray,
#                             asset_cov: np.ndarray):
#     """
#     Computes risk decomposition using marginal contibution to risk.
#     """
#     mctr = _numpy_mctr(portfolio_weights, asset_cov)
#     return mctr * portfolio_weights
#
#
# def _pandas_mctr_risk_decomp(portfolio_weights: pd.DataFrame,
#                              asset_cov: pd.DataFrame):
#     squeeze = False
#     if portfolio_weights.index.nlevels == 1:
#         squeeze = True
#         portfolio_weights = stack([portfolio_weights])
#
#     mctr_decomp = _numpy_mctr_risk_decomp(pandas_to_numpy(portfolio_weights),
#                                           pandas_to_numpy(asset_cov))
#     mctr_decomp = pd.DataFrame(np.row_stack(mctr_decomp),
#                                index=portfolio_weights.index,
#                                columns=portfolio_weights.columns)
#
#     if squeeze:
#         mctr_decomp = mctr_decomp.xs(0, level=1)
#
#     return mctr_decomp
#
#
# def mctr_risk_decomp(portfolio_weights, asset_cov):
#     array_like = [portfolio_weights, asset_cov]
#     if is_iterable_of(array_like, pd.DataFrame):
#         return _pandas_mctr_risk_decomp(portfolio_weights, asset_cov)
#     elif is_iterable_of(array_like, np.ndarray):
#         return _numpy_mctr_risk_decomp(portfolio_weights, asset_cov)
#     raise ValueError(
#         "portfolio_weights and asset_cov must either both be dataframes"
#         " or all be numpy arrays")
#
#
# def compute_risk_exposure():
#     pass
