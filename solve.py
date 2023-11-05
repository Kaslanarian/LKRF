import numpy as np
from project_simplex import project_simplex


def solve(v: np.ndarray, u: np.ndarray, rho: float, epsilon: float = 1e-8):
    '''
    Solve the problem:

    min_x  x^T v
    s.t.  \|x - u\|_2^2 \leq rho
         sum(x) = 1, x \geq 0
    '''
    max_lambda = np.inf
    min_lambda = 0.

    x = project_simplex(u)

    if np.linalg.norm(x - u)**2 > rho:
        raise ValueError("Problem is not feasible")

    start_lambda = 1.
    while np.isinf(max_lambda):
        x = project_simplex(u - v / start_lambda)
        lam_grad = (np.linalg.norm(x - u)**2 - rho) / 2

        if lam_grad < 0:
            max_lambda = start_lambda
        else:
            start_lambda *= 2

    while max_lambda - min_lambda > epsilon * start_lambda:
        lambda_ = (min_lambda + max_lambda) / 2
        x = project_simplex(u - v / lambda_)
        lam_grad = (np.linalg.norm(x - u)**2 - rho) / 2

        if lam_grad < 0:
            max_lambda = lambda_
        else:
            min_lambda = lambda_

    return x
