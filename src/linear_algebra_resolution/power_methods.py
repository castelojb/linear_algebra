import numpy as np


def regular_power_method(
    x: np.ndarray,
    u: np.ndarray = None,
    iterr_max: int = 100
) -> tuple[np.ndarray, float]:

    x = x.copy()

    u = np.ones(x.shape[0]) if u is None else u.copy()

    eigenvalue = 0

    for _ in range(iterr_max):

        u = x @ u

        eigenvalue = u.max()

        u = u / eigenvalue

    return u, eigenvalue


def inverse_power_method(
    x: np.ndarray,
    u: np.ndarray = None,
    iterr_max: int = 100
) -> tuple[np.ndarray, float]:
    x = x.copy()

    u = np.ones(x.shape[0]) if u is None else u.copy()

    eigenvalue = 0

    for _ in range(iterr_max):

        u = x / u

        eigenvalue = u.max()

        u = 1/eigenvalue*u

    return u, eigenvalue
