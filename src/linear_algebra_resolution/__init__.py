import numpy as np


def partial_pivotation(x_b: np.ndarray, center_idx: int) -> tuple[np.ndarray, bool]:
	n = x_b.shape[0]
	x_b = x_b.copy()

	sucess = False
	for rown in range(center_idx + 1, n):

		test_value = x_b[rown, center_idx]

		if not np.isclose(test_value, 0):
			x_b[center_idx, :], x_b[rown, :] = x_b[rown, :].copy(), x_b[center_idx, :].copy()
			sucess = True
			break

	return x_b, sucess


def total_pivotation(x_b: np.ndarray, center_idx: int, resp_arr_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
	n = x_b.shape[0]
	x_b = x_b.copy()
	resp_arr_idx = resp_arr_idx.copy()

	sucess = False
	for col in range(center_idx + 1, n):

		test_value = x_b[center_idx, col]

		if not np.isclose(test_value, 0):
			x_b[:, center_idx], x_b[:, col] = x_b[:, col].copy(), x_b[:, center_idx].copy()
			resp_arr_idx[center_idx], resp_arr_idx[col] = resp_arr_idx[col], resp_arr_idx[center_idx]
			sucess = True
			break

	return x_b, resp_arr_idx, sucess


def process_matrix_correction(x_b: np.ndarray, center_idx: int, resp_arr_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

	x_b, sucess = partial_pivotation(x_b, center_idx)

	if not sucess:
		x_b, resp_arr_idx, sucess = total_pivotation(x_b, center_idx, resp_arr_idx)

	if not sucess:
		err_txt = f"""\n{x_b} not exists valid value values to pivot number {center_idx}"""

		raise NameError(err_txt)

	return x_b, resp_arr_idx


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
