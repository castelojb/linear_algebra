import numpy as np


def reduced_row_echelon_form(x: np.ndarray) -> np.ndarray:

    x = x.copy()

    n, m = x.shape

    lead = 0

    for k in range(n):

        if lead >= m:
            return x

        i = k
        while np.isclose(x[i, lead], 0):
            i += 1

            if i == n:
                i = k
                lead += 1
                if m == lead:
                    return x

        x[i], x[k] = x[k], x[i]
        lv = x[k, lead]

        x[k] = x[k] / lv

        for i in range(n):
            if i != k:
                lv = x[i, lead]
                x[i] = x[i] - lv * x[k]
        lead += 1
    return x
