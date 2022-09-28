import numpy as np
from src.linear_algebra_resolution import process_matrix_correction, backward_substitution, History


class GaussHistory(History):
	def __init__(self, base_matrix):
		super().__init__(base_matrix)


def gauss_forward_elimination(x_b: np.ndarray, history: History) -> tuple[np.ndarray, np.ndarray, History]:

	x_b = x_b.copy()
	n = x_b.shape[0]

	resp_order = np.arange(n, dtype=int)

	for i in range(n):

		if np.isclose(x_b[i, i], 0):
			x_b, resp_order, history = process_matrix_correction(x_b, i, resp_order, history)

		for j in range(i + 1, n):
			alpha = x_b[j, i] / x_b[i, i]

			x_b = history.acum_operation_and_apply(
				history.multiply_and_subtract_rown,
				origin_rown_idx=i,
				dest_rown_idx=j,
				matrix=x_b.copy(),
				shape=n,
				coef=alpha,
			)

		x_b = history.acum_operation_and_apply(
			history.divide_line,
			origin_rown_idx=i,
			matrix=x_b,
			shape=n,
			coef=x_b[i, i],
		)

	return x_b, resp_order, history


def gaussian_elimination(x: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, History, np.ndarray]:

	history = GaussHistory(x)

	x_b = np.column_stack([x, b])

	x_forward, resp_order, history = gauss_forward_elimination(x_b, history)

	resp = backward_substitution(x_forward)

	return resp[resp_order], history, x_forward[:, :-1]
