import logging
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm

from pytrade.utils.pandas import stack, pandas_to_numpy

logger = logging.getLogger(__name__)


# TODO: use statsmodels/ linearmodels to fit HAR model instead of code below!
class CCCHAR:
    def __init__(self,
                 returns: pd.DataFrame,
                 realized_vol: pd.DataFrame,
                 *,
                 periods: Iterable[int] = (1, 7, 30),
                 min_periods: int = 100):
        self._returns = returns
        self._realized_vol = realized_vol
        self._periods = tuple(sorted(periods))

        self._models = []
        self._assets = returns.columns
        self._valid_assets = []

        last = []
        for asset in self._assets:
            if (~returns[asset].isnull()).sum() > min_periods:
                self._valid_assets.append(asset)
                x = sm.add_constant(
                    pd.concat([realized_vol[asset].rolling(x).mean().rename(f"rv_{x}")
                               for x in periods], axis=1))
                last.append(x.iloc[-1])

                x = x.shift()
                first_valid_index = x.iloc[:, -1].first_valid_index()
                self._models.append(sm.OLS(realized_vol[asset].loc[first_valid_index:],
                                           x.loc[first_valid_index:]))

        if last:
            self._last = pd.concat(last, axis=1, keys=self._valid_assets)
        self._fits = []
        self._fittedvalues = None
        self._std_resid = None
        self._corr = None

    def fit(self):
        if not self._valid_assets:
            return

        fitted_values = []
        std_resid = []
        for i, asset in enumerate(self._valid_assets):
            fit = self._models[i].fit()
            self._fits.append(fit)
            fitted_values.append(fit.fittedvalues)
            std_resid.append(self._returns[asset] / fit.fittedvalues)

        # mustn't use sort=False in unstack due to bug
        # (see: https://github.com/pandas-dev/pandas/issues/54987)
        self._fittedvalues = stack(
            fitted_values,
            self._valid_assets,
            names="model",
        ).unstack().reindex(columns=self._valid_assets)
        self._std_resid = stack(
            std_resid, self._valid_assets, names="model",
        ).unstack().reindex(columns=self._valid_assets)
        self._corr = self._std_resid.corr()

    @property
    def fitted_values(self):
        if self._fittedvalues is not None:
            return self._fittedvalues.reindex(columns=self._assets)

    def forecast(self):
        if self._valid_assets:
            cond_vol = []
            for i, asset in enumerate(self._valid_assets):
                cond_vol.append(self._fits[i].predict(self._last[asset])[0])
            cond_vol = np.diag(np.array(cond_vol))
            cov_matrix = cond_vol @ self._corr.values @ cond_vol
            return pd.DataFrame(cov_matrix, index=self._valid_assets,
                                columns=self._valid_assets).reindex(
                index=self._assets, columns=self._assets)
        return pd.DataFrame(np.nan, index=self._assets, columns=self._assets)


def compute_ccc_har_cov(
        returns: pd.DataFrame,
        realized_vol: pd.DataFrame,
        *,
        periods: Iterable[int] = (1, 7, 30),
        min_periods: int = 100,
        show_progress: bool = False
):
    asset_cov = []
    for date in tqdm(returns.index, disable=(not show_progress)):
        model = CCCHAR(returns.loc[:date], realized_vol.loc[:date], periods=periods,
                       min_periods=min_periods)
        model.fit()
        asset_cov.append(model.forecast())
    return stack(asset_cov, returns.index, sort_level=1,
                 sort_remaining=False).swaplevel(0, 1)


# TODO: better arg name than overlap?
class HARCholeskyRiskModel:

    def __init__(self, lags: Iterable[int] = (1, 5, 20), overlap: bool = False):
        self._lags = tuple(sorted(lags))
        self._overlap = overlap
        self._fit = None
        self._assets = None
        self._valid_assets = None
        self._invalid_mask = None

        self._offsets = (0,) * len(self._lags)
        if not overlap:
            self._offsets = tuple(np.cumsum([0] + list(self._lags[:-1])))

    def fit(self, realized_cov: pd.DataFrame) -> "HARCholeskyRiskModel":
        self._assets = realized_cov.columns

        # ignore assets with all nan/ zero covariances at any timestep since cholesky
        # decomposition can't be computed if so
        invalid_mask = realized_cov.isnull().groupby(level=0).all().any()
        invalid_mask = invalid_mask | (realized_cov == 0).groupby(level=0).all().any()
        self._invalid_mask = invalid_mask

        invalid_assets = invalid_mask[invalid_mask].index
        if len(invalid_assets) > 0:
            logger.debug(f"Excluding assets from fit due to all nan/ zero columns:"
                         f" {', '.join(invalid_assets)}")
        if invalid_mask.all():
            raise ValueError("Error fitting model; no valid assets")

        valid_assets = self._assets[~invalid_mask]
        self._valid_assets = valid_assets

        # must sort_index below since loc changes time ordering
        idx = pd.IndexSlice
        realized_cov = realized_cov.loc[
            idx[:, valid_assets], valid_assets].sort_index(
            level=0, sort_remaining=False)
        dcmp = self._compute_cholesky_dcmp(realized_cov)

        X = self._create_design_matrix(dcmp)
        y = dcmp.stack(level=(0, 1), future_stack=True)

        mod = sm.OLS(
            y.groupby(level=(1, 2)).shift(-1), X, missing="drop"
        )
        self._fit = mod.fit()
        return self

    @staticmethod
    def _compute_cholesky_dcmp(realized_cov: pd.DataFrame) -> pd.DataFrame:
        times = realized_cov.index.unique(level=0)
        assets = realized_cov.columns

        tril = np.tril_indices(len(assets))
        dcmp = np.linalg.cholesky(pandas_to_numpy(realized_cov))  # T x N x N
        dcmp = dcmp[:, tril[0], tril[1]]  # T x (N (N+1) / 2)

        return pd.DataFrame(
            dcmp,
            index=times,
            columns=pd.MultiIndex.from_arrays(
                [assets[tril[0]], assets[tril[1]]],
                names=("asset_1", "asset_2")
            ),
        )

    def _create_design_matrix(self, cholesky_dcmp: pd.DataFrame):
        dcmp = cholesky_dcmp
        # don't have to specify min periods in rolling since no nans
        X = stack([dcmp.rolling(l).mean().shift(o) for l, o in
                   zip(self._lags, self._offsets)], self._lags)
        # mustn't use sort=False in unstack below due to bug with multi-indexes!
        X = X.stack(level=(0, 1), future_stack=True).unstack(level=1)
        X["idx"] = X.index.get_level_values(1) + "/" + X.index.get_level_values(2)
        X = pd.get_dummies(X, columns=["idx"], dtype=int)
        # TODO: re-order dummy columns in X as well as index
        index = dcmp.stack(level=(0, 1), future_stack=True).index
        return X.reindex(index=index)

    def predict(self, realized_cov: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts covariance.

        Parameters
        ----------
        realized_cov
            Realized covariance. Must have at least as many rows as the maximum
            lag used to fit the model.

        Returns
        -------
        Forecast covariance matrix.
        """
        if self._fit is None:
            raise ValueError("Error making predictions; model hasn't been fit")

        N = len(self._valid_assets)
        tril = np.tril_indices(len(self._valid_assets))
        times = realized_cov.index.unique(level=0)

        idx = pd.IndexSlice
        realized_cov = realized_cov.loc[
            idx[:, self._valid_assets], self._valid_assets].sort_index(
            level=0, sort_remaining=False)
        dcmp = self._compute_cholesky_dcmp(realized_cov)
        X = self._create_design_matrix(dcmp)

        asset_cov = np.zeros((len(times), N, N))
        asset_cov[:, tril[0], tril[1]] = self._fit.predict(X).values.reshape(
            (len(times), len(tril[0])))
        asset_cov = np.einsum("tik,tjk->tij", asset_cov, asset_cov)
        asset_cov = pd.DataFrame(
            np.row_stack(asset_cov), index=pd.MultiIndex.from_product(
                [times, self._valid_assets]), columns=self._valid_assets
        )
        return asset_cov.reindex(
            index=pd.MultiIndex.from_product([times, self._assets]),
            columns=self._assets)


def fit_har_cholesky_risk_model(realized_cov: pd.DataFrame,
                                lags: Iterable[int] = (1, 5, 20),
                                overlap: bool = False):
    return HARCholeskyRiskModel(lags, overlap=overlap).fit(realized_cov)


# TODO: allow other exogeneous variables
def compute_rolling_har_cholesky_asset_cov(realized_cov: pd.DataFrame,
                                           lags: Iterable[int] = (1, 5, 20),
                                           overlap: bool = False,
                                           window: int = 200,
                                           show_progress: bool = False):
    max_lag = max(lags) if overlap else np.sum(list(lags))
    window += max_lag

    index = realized_cov.index
    assets = realized_cov.columns

    first_valid_index = realized_cov.first_valid_index()
    first_valid_time = first_valid_index[0]
    realized_cov = realized_cov.loc[first_valid_time:]

    times = realized_cov.index.unique(level=0)

    asset_covs = {}
    for i in tqdm(range(len(times) - window + 1), disable=(not show_progress)):
        curr_idx = i + window - 1
        curr_time = times[curr_idx]
        realized_cov_slice = realized_cov.loc[times[i]:curr_time]
        model = fit_har_cholesky_risk_model(realized_cov_slice, lags, overlap=overlap)
        # only pass data needed to make prediction for curr_time
        asset_covs[curr_time] = model.predict(
            realized_cov_slice.loc[times[curr_idx - max_lag + 1]:]).loc[curr_time]

    asset_cov = pd.concat(asset_covs.values(), keys=asset_covs.keys(), axis=0)
    return asset_cov.reindex(index=index, columns=assets)
