import numpy as np
from src.linear_algebra_resolution import process_matrix_correction, backward_substitution


def forward_elimination(x_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

	x_b = x_b.copy()
	n = x_b.shape[0]

	resp_order = np.arange(n, dtype=int)

	for i in range(n):

		if np.isclose(x_b[i, i], 0):
			x_b, resp_order = process_matrix_correction(x_b, i, resp_order)

		for j in range(i + 1, n):
			alpha = x_b[j, i] / x_b[i, i]

			for k in range(n + 1):
				x_b[j, k] = x_b[j, k] - alpha * x_b[i, k]

	return x_b, resp_order


def gaussian_elimination(x: np.ndarray, b: np.ndarray) -> np.ndarray:

	x_b = np.column_stack([x, b])

	x_forward, resp_order = forward_elimination(x_b)

	resp = backward_substitution(x_forward)

	return resp[resp_order]
