import pytest
import numpy as np

from pytrade.stats.lm import compute_t_and_p_values


@pytest.mark.parametrize(
    ["X", "y", "coef_", "omega", "tvalues", "pvalues"],
    [
        pytest.param(
            np.array([[-0.42, -0.06],
                      [-2.14, 1.64],
                      [-1.79, -0.84],
                      [0.5, -1.25],
                      [-1.06, -0.91],
                      [0.55, 2.29],
                      [0.04, -1.12],
                      [0.54, -0.6],
                      [-0.02, 1.18],
                      [-0.75, 0.01],
                      [-0.88, -0.16],
                      [0.26, -0.99],
                      [-0.34, -0.24],
                      [-0.64, -1.19],
                      [-1.42, -0.15],
                      [-0.27, 2.23],
                      [-2.43, 0.11],
                      [0.37, 1.36],
                      [0.5, -0.84],
                      [0., 0.54]]),
            np.array([-0.02, -0.12, -0.13, 0.12, 0.04, -0.04, 0.07, 0.04, -0.1,
                      -0.03, 0.07, -0.01, -0.06, -0.02, -0., 0.15, -0.23, 0.08,
                      0.08, -0.38]),
            np.array([0.05092351, -0.01629069]),
            np.array([0.0001, 0.0004, 0.0009, 0.0016, 0.0025, 0.0036, 0.0049, 0.0064,
                      0.0081, 0.01, 0.0121, 0.0144, 0.0169, 0.0196, 0.0225, 0.0256,
                      0.0289, 0.0324, 0.0361, 0.04]),
            np.array([6.0011891, -1.51687984]),
            np.array([1.12425596e-05, 1.46663316e-01]),
        )
    ]
)
def test_compute_realized_cov(X, y, coef_, omega, tvalues, pvalues):
    weights = 1.0 / omega
    actual = compute_t_and_p_values(X, y, coef_, weights=weights)
    np.testing.assert_allclose(actual[0], tvalues, atol=1e-3, rtol=0)
    np.testing.assert_allclose(actual[1], pvalues, atol=1e-3, rtol=0)
