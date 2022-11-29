import numpy as np

from src.linear_algebra_resolution.gaussian_elimination import gaussian_elimination
from src.linear_algebra_resolution.lu_decomposition import lu_resolution


def regular_power_method(
    a: np.ndarray,
    x: np.ndarray = None,
    eps: float = 0.0001,
    iterr_max: int = 100
) -> tuple[np.ndarray, float, bool]:

    a = a.copy()

    x = np.ones(a.shape[0]) if x is None else x.copy()

    converged = False

    eigenvalue = 0
    last_eigenvalue = eigenvalue

    for _ in range(iterr_max):

        x_hat = x / np.linalg.norm(x)

        x = a @ x_hat

        eigenvalue = x_hat @ x

        if abs((eigenvalue - last_eigenvalue) / eigenvalue) < eps:
            converged = True
            break

        last_eigenvalue = eigenvalue

    x_hat = x / np.linalg.norm(x)

    return x_hat, eigenvalue, converged


def inverse_power_method(
    a: np.ndarray,
    x: np.ndarray = None,
    eps: float = 0.0001,
    iterr_max: int = 100
) -> tuple[np.ndarray, float, bool]:

    a_pinv = np.linalg.pinv(a)

    x_hat, eigenvalue, converged = regular_power_method(a_pinv, x, eps=eps, iterr_max=iterr_max)

    inv_eigenvalue = 1/eigenvalue

    return x_hat, inv_eigenvalue, converged
