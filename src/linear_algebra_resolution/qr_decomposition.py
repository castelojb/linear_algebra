import itertools

import numpy as np

from src.linear_algebra_resolution.householder_method import householder
from src.linear_algebra_resolution.power_methods import regular_power_method


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


def make_vector_under_principal_dialg(d: np.ndarray) -> np.ndarray:
    vectors = np.concatenate(
        [
            d[i + 1:, i]
            for i in range(d.shape[0])
        ]
    )

    return vectors


def simetric_convergence(l: np.ndarray, l_past: np.ndarray, eps: float) -> bool:

    return np.linalg.norm(l, 2) < eps


def not_simetric_convergence(l: np.ndarray, l_past: np.ndarray, eps: float) -> bool:

    dist = np.linalg.norm(l - l_past)
    return abs(dist) < eps


def extract_bts(d: np.ndarray) -> list[np.ndarray | float] | tuple[list[np.ndarray | float], list[int]]:
    bts = []

    idx = 0
    while idx <= d.shape[0]-1:

        if idx == d.shape[0]-1 or np.isclose(d[idx+1, idx], 0):
            bts.append(d[idx, idx])
            idx += 1
            continue

        bts.append(np.array([
            [d[idx, idx], d[idx, idx+1]],
            [d[idx+1, idx], d[idx+1, idx + 1]]
        ]))

        idx += 2

    return bts


def find_roots(a: float, b: float, c: float) -> tuple[float, float] | list[float] | \
                                               tuple[np.complex, np.complex]:
    dis_form = b * b - 4 * a * c
    sqrt_val = np.sqrt(np.abs(dis_form))

    if dis_form > 0:
        return ((-b + sqrt_val) / (2 * a)), ((-b - sqrt_val) / (2 * a))

    elif dis_form == 0:
        return [-b / (2 * a)]

    else:
        return (- b / (2 * a)) + 1j, (- b / (2 * a)) - 1j


def block_elimination(B, y=np.zeros((2, 1))):
    pivot = B[0, 0]
    pivot_conj = np.conjugate(pivot)
    B_01 = (B[0, 1] * pivot_conj) / (pivot * pivot_conj)
    y_0 = (y[0, 0] * pivot_conj) / (pivot * pivot_conj)
    B_11 = B[1, 0] * B_01 - B[1, 1]
    y_1 = B[1, 0] * y_0 - y[1, 0]
    if np.isclose(B_11, 0):
        x1 = y_0 - B_01
        x2 = 1
    else:
        x2 = y_1 / B_11
        x1 = y_0 - B_01 * x2
    return np.array([x1, x2])


def not_simetric_extract_eignvalues(d: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bts = extract_bts(d)

    eignvalues = [

        find_roots(
            1,
            block[0, 0] + block[1, 1],
            (block[0, 0]*block[1, 1]) - (block[0, 1]*block[1, 0])
        ) if not isinstance(block, float) else [block] for block in bts

    ]

    flatlist_eignvalues = np.array(list(itertools.chain.from_iterable(eignvalues)))
    resps = []
    for eignvalue in flatlist_eignvalues:

        d_ = d - (eignvalue * np.eye(d.shape[0]))

        resp, _, _ = regular_power_method(d_)

        resps.append(resp)

    return flatlist_eignvalues, np.vstack(resps)


def simetric_extract_eignvalues(d: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.diag(d), p


def qr_decomposition(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    n = a.shape[0]
    ab = a.copy()
    q = np.eye(n)

    for c in range(n-1):

        hc = householder_matrix(ab, c)

        ab = hc @ ab

        q = q @ hc

    return q, ab


def qr_method(a: np.ndarray, eps: float = 000.1) -> tuple[np.ndarray, np.ndarray]:

    d = a.copy()

    # a_hat, h = householder(a)
    # p = h.copy()
    p = np.eye(a.shape[0])
    is_simetric = np.isclose(a, a.T).all()

    convergence_function = simetric_convergence if is_simetric else not_simetric_convergence
    eig_extractor = simetric_extract_eignvalues if is_simetric else not_simetric_extract_eignvalues

    l_past = make_vector_under_principal_dialg(d)

    while True:

        q, r = qr_decomposition(d)

        d = r @ q

        p = p @ q

        l = make_vector_under_principal_dialg(d)
        if convergence_function(l, l_past, eps):
            break

        l_past = l.copy()

    return eig_extractor(d, p)
