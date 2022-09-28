import numpy as np

from src.linear_algebra_resolution import History, backward_substitution
from src.linear_algebra_resolution.gaussian_elimination import gauss_forward_elimination


class GaussJordamHistory(History):
	def __init__(self, base_matrix):
		super().__init__(base_matrix)


def jordam_backward_elimination(x_b: np.ndarray, history: History) -> tuple[np.ndarray, History]:

	x_b = x_b.copy()
	n = x_b.shape[0]

	for i in range(n-1, -1, -1):

		for j in range(i-1, -1, -1):
			alpha = x_b[j, i] / x_b[i, i]

			x_b = history.acum_operation_and_apply(
				history.multiply_and_subtract_rown,
				origin_rown_idx=i,
				dest_rown_idx=j,
				matrix=x_b,
				shape=n,
				coef=alpha,
			)

	return x_b, history


def gaussian_jordam_elimination(x: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, History]:

	history = GaussJordamHistory(x)

	x_b = np.column_stack([x, b])

	x_forward, resp_order, history = gauss_forward_elimination(x_b, history)

	x_bacward, history = jordam_backward_elimination(x_forward, history)

	resp = backward_substitution(x_bacward)

	return resp[resp_order], history
