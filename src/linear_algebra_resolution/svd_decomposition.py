import numpy as np

from src.linear_algebra_resolution.power_methods import regular_power_method


def svd(a, k=None, eps=1e-10):

    a = a.astype(float)
    a_ = a.copy()
    n, m = a.shape

    svd_so_far = []

    k = min(n, m) if k is None else k

    for i in range(k):

        if n > m:
            v, sigma, _ = regular_power_method(a_, eps=eps)
            u_unnormalized = a @ v
            u = u_unnormalized / sigma
        else:
            u, sigma, _ = regular_power_method(a_, eps=eps)
            v_unnormalized = a.T @ u
            v = v_unnormalized / sigma

        a_ -= sigma * np.outer(u, v)

        svd_so_far.append((sigma, u, v))

    singular_values, us, vs = [np.array(x) for x in zip(*svd_so_far)]

    return np.diag(singular_values), us.T, vs
