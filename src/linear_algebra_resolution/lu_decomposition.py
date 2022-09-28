import numpy as np
from src.linear_algebra_resolution import History, backward_substitution


def lu_decomposition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

	x = x.copy()

	n = x.shape[0]
	u = np.eye(n)
	l = np.eye(n)

	for k in range(n):

		if np.isclose(x[k,k], 0):
			err_txt = f"""\n{x} not exists valid value values to pivot number {k}"""
			raise NameError(err_txt)

		u[k, k] = x[k, k]

		for i in range(k+1, n):
			l[i, k] = x[i, k] / u[k, k]
			print(l[i, k])

			u[k, i] = x[k, i]

		for i in range(k+1, n):
			for j in range(k+1, n):
				x[i, j] = x[i, j] - l[i, k] * u[k, j]

	return l, u


def lu_resolution(x: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

	l, u = lu_decomposition(x)

	u_b = np.column_stack([u, b])

	resp = backward_substitution(u_b)

	return l, u, resp
