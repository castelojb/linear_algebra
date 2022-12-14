import numpy as np


def sor_solver(
    a: np.ndarray,
    b: np.ndarray,
    omg: float,
    phi=None,
    eps=1e-10
) -> np.ndarray:
    phi = phi if phi is not None else np.zeros(b.shape[0])

    res = np.linalg.norm(
        (a @ phi) - b
    )

    while res > eps:

        for i in range(a.shape[0]):

            sigma = 0

            for j in range(a.shape[1]):

                if j != i:
                    sigma += a[i, j] * phi[j]

            phi[i] = (1 - omg) * phi[i] + (omg / a[i, i]) * (b[i] - sigma)

        res = np.linalg.norm(
          (a @ phi) - b
        )

    return phi
