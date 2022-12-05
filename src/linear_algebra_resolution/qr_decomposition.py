import numpy as np


def householder_matrix(a: np.ndarray, c: int) -> np.ndarray:

    n = a.shape[0]

    v = np.zeros([n, 1])
    v1 = np.zeros([n, 1])
    eye = np.eye(n)

    v[c:n, 0] = a[c:n, c]

    lv = np.linalg.norm(v, 2)

    v1[c] = lv

    n_vector = v - v1

    vn = n_vector / np.linalg.norm(n_vector, 2)

    h = eye - 2 * vn @ vn.T

    return h


def qr_decomposition(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    n = a.shape[0]
    ab = a.copy()
    q = np.eye(n)

    for c in range(n-1):

        hc = householder_matrix(ab, c)

        ab = hc @ ab

        q = q @ hc

    return q, ab


def check_len(d: np.ndarray):

    vectors = np.concatenate([
        d[i+1:, i]
        for i in range(d.shape[0])
    ])

    return np.linalg.norm(vectors, 2)


def qr_method(a: np.ndarray, h: np.ndarray, eps: float = 000.1, iter_max=1000) -> tuple[np.ndarray, np.ndarray]:

    d = a.copy()
    p = h.copy()

    i = 0
    while True:

        q, r = qr_decomposition(d)

        d = r @ q

        p = p @ q

        l = check_len(d)
        if l < eps:
            break

    return d, p
