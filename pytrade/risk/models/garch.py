# suppress arch warnings
import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd
from arch.univariate import arch_model
from tqdm import tqdm

from pytrade.utils.pandas import stack

warnings.filterwarnings("ignore")


class CCCGarch:
    def __init__(self,
                 returns: pd.DataFrame,
                 *,
                 mean: Literal[
                     "Constant", "Zero", "LS", "AR", "ARX", "HAR", "HARX", "constant",
                     "zero"] = "Zero",
                 vol: Literal[
                     "GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH", "FIGARCH"
                 ] = "GARCH",
                 p: int = 1,
                 o: int = 0,
                 q: int = 1,
                 power: float = 2.0,
                 dist: Literal[
                     "normal",
                     "gaussian",
                     "t",
                     "studentst",
                     "skewstudent",
                     "skewt",
                     "ged",
                     "generalized error",
                 ] = "normal"):
        self._returns = returns

        self._models = []
        self._assets = returns.columns

        valid_mask = returns.isnull().sum() == 0
        self._valid_assets = valid_mask[valid_mask].index.tolist()
        for asset in self._valid_assets:
            # must set rescale=True to avoid convergence warnings!
            self._models.append(
                arch_model(returns[asset].dropna(), mean=mean, vol=vol, p=p, o=o,
                           q=q, power=power, dist=dist, rescale=True))

        self._fits = []
        self._params = None
        self._cond_vol = None
        self._resid = None
        self._std_resid = None
        self._corr = None

    def fit(self, starting_values: Optional[pd.DataFrame] = None):
        if not self._valid_assets:
            return

        params = []
        cond_vol = []
        resid = []
        std_resid = []
        for i, asset in enumerate(self._valid_assets):
            model_starting_values = starting_values[asset].rename("params") if (
                    starting_values is not None and asset in
                    starting_values) else None
            fit = self._models[i].fit(update_freq=0, disp="off",
                                      options={"maxiter": 1000},
                                      starting_values=model_starting_values)
            self._fits.append(fit)
            # must divide by fit.scale below!
            cond_vol_ = fit.conditional_volatility / fit.scale
            params.append(fit.params)
            cond_vol.append(cond_vol_)
            resid.append(fit.resid)
            std_resid.append(fit.resid / cond_vol_)

        # so we can pass params as starting values, row order must match that of
        # fit.params
        self._params = stack(
            params, self._valid_assets, names="model", sort_level=None,
        ).unstack()
        self._cond_vol = stack(
            cond_vol,
            self._valid_assets,
            names="model",
            sort_remaining=False,
        ).unstack(sort=False)
        self._resid = stack(
            resid, self._valid_assets, names="model", sort_remaining=False
        ).unstack(sort=False)
        self._std_resid = stack(
            std_resid, self._valid_assets, names="model", sort_remaining=False
        ).unstack(sort=False)
        self._corr = self._std_resid.corr()

    @property
    def params(self):
        if self._params is not None:
            return self._params.reindex(columns=self._assets)

    @property
    def cond_vol(self):
        if self._cond_vol is not None:
            return self._cond_vol.reindex(columns=self._assets)

    @property
    def resid(self):
        if self._resid is not None:
            return self._resid.reindex(columns=self._assets)

    @property
    def std_resid(self):
        if self._std_resid is not None:
            return self._std_resid.reindex(columns=self._assets)

    @property
    def corr(self):
        if self._corr is not None:
            return self._corr.reindex(index=self._assets, columns=self._assets)

    def forecast(self):
        if self._valid_assets:
            cond_var = []
            for i, asset in enumerate(self._valid_assets):
                fit = self._fits[i]
                cond_var_ = fit.forecast(horizon=1).variance.values[-1][
                                0] / fit.scale ** 2
                cond_var.append(cond_var_)
            cond_vol = np.diag(np.sqrt(np.array(cond_var)))
            cov_matrix = cond_vol @ self._corr.values @ cond_vol
            return pd.DataFrame(cov_matrix, index=self._valid_assets,
                                columns=self._valid_assets).reindex(
                index=self._assets, columns=self._assets)
        return pd.DataFrame(np.nan, index=self._assets, columns=self._assets)


def compute_ccc_garch_cov(
        returns: pd.DataFrame,
        window: int,
        *,
        mean: Literal[
            "Constant", "Zero", "LS", "AR", "ARX", "HAR", "HARX", "constant",
            "zero"] = "Constant",
        vol: Literal[
            "GARCH", "ARCH", "EGARCH", "FIGARCH", "APARCH", "HARCH", "FIGARCH"
        ] = "GARCH",
        p: int = 1,
        o: int = 0,
        q: int = 1,
        power: float = 2.0,
        dist: Literal[
            "normal",
            "gaussian",
            "t",
            "studentst",
            "skewstudent",
            "skewt",
            "ged",
            "generalized error",
        ] = "normal",
        show_progress: bool = False,

):
    asset_cov = {}
    times = returns.index

    for i in tqdm(range(window - 1, len(times)), disable=(not show_progress)):
        time = times[i]
        slice_ = returns.iloc[i + 1 - window:i + 1]
        model = CCCGarch(slice_, mean=mean, vol=vol, p=p, o=o,
                         q=q, power=power, dist=dist)
        # TODO: optionally pass starting_values below?
        model.fit(starting_values=None)
        asset_cov[time] = model.forecast()

    asset_cov = stack(asset_cov, sort_level=1, sort_remaining=False).swaplevel(0, 1)
    return asset_cov.reindex(
        pd.MultiIndex.from_product([times, returns.columns]))
