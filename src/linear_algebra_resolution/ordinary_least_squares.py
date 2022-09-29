import numpy as np


def least_squares(x: np.ndarray, y: np.ndarray, l2_regulazation=1) -> np.ndarray:

	l2_reg_matrix = np.eye(x.shape[1]) * l2_regulazation

	w = np.linalg.inv((x.T @ x) + l2_reg_matrix) @ x.T @ y

	return w
