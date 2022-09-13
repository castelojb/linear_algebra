import numpy as np

from src.linear_algebra_resolution._history import History


def partial_pivotation(x_b: np.ndarray, center_idx: int, history: History) -> tuple[np.ndarray, bool, History]:
	n = x_b.shape[0]
	x_b = x_b.copy()

	sucess = False
	for rown in range(center_idx + 1, n):

		test_value = x_b[rown, center_idx]

		if not np.isclose(test_value, 0):
			x_b = history.acum_operation_and_apply(
				history.permutation_rown,
				origin_rown_idx=center_idx,
				dest_rown_idx=rown,
				matrix=x_b,
				shape=n
			)
			sucess = True
			break

	return x_b, sucess, history


def total_pivotation(x_b: np.ndarray, center_idx: int, resp_arr_idx: np.ndarray, history: History) -> tuple[np.ndarray, np.ndarray, bool, History]:
	n = x_b.shape[0]
	x_b = x_b.copy()
	resp_arr_idx = resp_arr_idx.copy()

	sucess = False
	for col in range(center_idx + 1, n):

		test_value = x_b[center_idx, col]

		if not np.isclose(test_value, 0):

			x_b, resp_arr_idx = history.acum_operation_and_apply(
				history.permutation_col,
				origin_rown_idx=center_idx,
				dest_rown_idx=col,
				idx_arr=resp_arr_idx,
				matrix=x_b,
				shape=n
			)
			sucess = True
			break

	return x_b, resp_arr_idx, sucess, history


def process_matrix_correction(x_b: np.ndarray, center_idx: int, resp_arr_idx: np.ndarray, history: History) -> tuple[np.ndarray, np.ndarray, History]:

	x_b, sucess, history = partial_pivotation(x_b, center_idx, history)

	if not sucess:
		x_b, resp_arr_idx, sucess, history = total_pivotation(x_b, center_idx, resp_arr_idx, history)

	if not sucess:
		err_txt = f"""\n{x_b} not exists valid value values to pivot number {center_idx}"""

		raise NameError(err_txt)

	return x_b, resp_arr_idx, history


def backward_substitution(x_b: np.ndarray) -> np.ndarray:
	n = x_b.shape[0]

	x = np.empty(n)

	x[n - 1] = x_b[n - 1, n] / x_b[n - 1, n - 1]

	for i in range(n - 2, -1, -1):
		x[i] = x_b[i, n]

		for j in range(i + 1, n):
			x[i] = x[i] - x_b[i, j] * x[j]

		x[i] = x[i] / x_b[i, i]

	return x
