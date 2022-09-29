import numpy as np

def cholesky_decomposition(x: np.ndarray) -> np.ndarray:

    x = x.copy()
    n = x.shape[0]

    g = np.zeros_like(x)

    for k in range(n):

        if x[k, k] <=0:
            err_txt = f"""\n{x} not is positive defined"""
            raise NameError(err_txt)

        g[k, k] = np.sqrt(x[k, k])
        g[k, k + 1:] = x[k, k + 1:] / g[k, k]

        for j in range(k + 1, n):
            x[j, j:] = x[j, j:] - g[k, j] * g[k, j:]

    return g.T
