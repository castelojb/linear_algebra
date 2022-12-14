import numpy as np


def conjugate_gradient(
    a: np.ndarray,
    b: np.ndarray,
    x0=None,
    eps=1e-5,
    maxiter=100
):
    x0 = x0 if x0 is not None else np.zeros(b.shape[0])

    grad0 = (a @ x0) - b

    d = -grad0

    for _ in range(maxiter):
        alpha = (grad0.T @ grad0) / (d.T @ a @ d)

        x0 = x0 + d * alpha

        gradi = grad0 + ((a * alpha) @ d)

        if np.linalg.norm(gradi) < eps:
            return x0

        betai = (gradi.T @ gradi) / (grad0.T @ grad0)

        d = - gradi + betai * d
        grad0 = gradi

    return x0
