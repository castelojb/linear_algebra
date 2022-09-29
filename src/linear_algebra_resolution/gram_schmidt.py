import numpy as np

from src.linear_algebra_resolution.cholesky_decomposition import cholesky_decomposition


def gram_schmidt_ortogonalization(x: np.ndarray) -> np.ndarray:

    x = x.copy()
    n = x.shape[1]

    for j in range(n):
        for k in range(j):
            x[:, j] -= np.dot(x[:, k], x[:, j]) * x[:, k]

        x[:, j] = x[:, j] / np.linalg.norm(x[:, j])

    return x


def gram_schmidt_from_cholesky(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    g = cholesky_decomposition(x)

    out = gram_schmidt_ortogonalization(g)

    return g, out
