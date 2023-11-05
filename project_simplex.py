import numpy as np


def project_simplex(v: np.ndarray, B: float = 1.):
    u = np.sort(v)[::-1]
    sv = np.cumsum(u)
    judge = (u > (sv - B) / np.arange(1, len(v) + 1)).astype(int)
    rho = np.nonzero(judge)[0][-1]
    theta = (sv[rho] - B) / (rho + 1)
    return np.maximum(v - theta, 0)
