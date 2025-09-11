import numpy as np
from scipy.linalg import orth


def svd(a):
    u, s, v = np.linalg.svd(a, full_matrices=False)
    s = np.diag(s)
    v = v.T
    # multiply each column of u and v in such a way that first row of v is
    # all positive (it's ok to flip the sign of a column in v so long as the
    # sign of the corresponding column in u is also flipped)
    sign = np.sign(v[0, :])
    u *= sign
    v *= sign
    return u, s, v


def compute_orth_basis(a, full_matrices=True):
    basis = orth(a)
    if full_matrices and (basis.shape[1] < a.shape[1]):
        basis = np.pad(basis, ((0, 0), (0, a.shape[1] - basis.shape[1])))
    return basis
