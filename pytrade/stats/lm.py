from typing import Tuple, Optional

import numpy as np
from scipy.stats import t


def compute_t_and_p_values(X: np.ndarray, y: np.ndarray, coef: np.ndarray,
                           weights: Optional[np.ndarray] = None) -> Tuple[
    np.ndarray, np.ndarray]:
    N, K = X.shape

    if weights is None:
        weights = np.ones(N)

    resids = y - X @ coef
    sigma_hat = np.sqrt(np.sum(weights * np.square(resids)) / (N - K))
    X = X * np.sqrt(weights[:, np.newaxis])
    beta_cov = np.linalg.inv(X.T @ X)
    t_values = coef / (sigma_hat * np.sqrt(np.diagonal(beta_cov)))
    p_values = t.sf(np.abs(t_values), N - K) * 2
    return t_values, p_values


def compute_r2(X: np.ndarray, y: np.ndarray, coef: np.ndarray,
               weights: Optional[np.ndarray] = None, adjust: bool = False) -> float:
    N, K = X.shape

    if weights is None:
        weights = np.ones(N)

    resids = y - X @ coef
    # remove np.mean(y) below for uncentered r2
    r2 = 1 - np.sum(weights * resids ** 2) / np.sum(weights * (y - np.mean(y)) ** 2)
    if adjust:
        return 1 - (1 - r2) * (N - 1) / (N - K - 1)
    return r2


def lm(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    t_values, p_values = compute_t_and_p_values(X, y, coef)
    return coef, t_values, p_values
