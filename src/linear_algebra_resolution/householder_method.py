import numpy as np


def householder_matrix(a: np.ndarray, c: int) -> np.ndarray:

    n = a.shape[0]

    v = np.zeros([n, 1])
    v1 = np.zeros([n, 1])
    eye = np.eye(n)

    v[c+1:n, 0] = a[c+1:n, c]

    lv = np.linalg.norm(v, 2)

    v1[c+1] = lv

    n_vector = v - v1
    vn = n_vector / np.linalg.norm(n_vector)

    h = eye - 2 * vn @ vn.T

    return h


def householder(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    n = a.shape[0]

    ab = a.copy()
    
    h = np.eye(n)
    
    for c in range(n-2):
        
        hc = householder_matrix(ab, c)

        ab = hc @ ab @ hc
        h = h @ hc

    return ab, h
