import numpy as np
from src.linear_algebra_resolution import History, backward_substitution


def lu_decomposition(x_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

	x_b = x_b.copy()

	n = x_b.shape[0]
	u = np.eye(n)
	l = np.eye(n)

	for k in range(n):

		if np.isclose(x_b[k, k], 0):
			err_txt = f"""\n{x_b} not exists valid value values to pivot number {k}"""
			raise NameError(err_txt)

		u[k, k] = x_b[k, k]

		for i in range(k+1, n):
			alpha = x_b[i, k] / u[k, k]

			l[i, k] = alpha
			u[k, i] = x_b[k, i]

			x_b = History.multiply_and_subtract_rown(
				origin_rown_idx=k,
				dest_rown_idx=i,
				matrix=x_b.copy(),
				shape=n,
				coef=alpha
			)

	return l, u, x_b


def lu_resolution(x: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

	x_b = np.column_stack([x, b])

	l, u, x_b = lu_decomposition(x_b)

	u_b = np.column_stack([u, x_b[:, -1]])

	resp = backward_substitution(u_b)

	return l, u, resp
