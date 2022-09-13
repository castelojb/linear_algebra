from functools import reduce
from typing import Any

import numpy as np
from typing import Callable


class History:

	def __init__(self, base_matrix):
		self.base_matrix = base_matrix
		self.operations_matrix = []

	def acum_operation(self, function: Callable[[Any], np.ndarray], **kwargs):

		kwargs['history_matrix'] = True

		op_matrix = function(**kwargs)

		if (op_matrix != np.eye(self.base_matrix.shape[0])).any():

			self.operations_matrix.append(
				op_matrix
			)

		return self.operations_matrix

	def acum_operation_and_apply(self, function: Callable[[Any], np.ndarray], **kwargs):
		out = function(**kwargs)
		self.acum_operation(function, **kwargs)

		return out

	def get_acum_op_matrix(self) -> np.ndarray:

		return reduce(lambda a, b: b @ a, self.operations_matrix)

	def apply_operations_matrix(self):

		acum_matrix = self.get_acum_op_matrix()

		return acum_matrix @ self.base_matrix

	@staticmethod
	def process_matrix(history_matrix: bool, matrix: np.ndarray | None, shape: int) -> np.ndarray:

		if history_matrix:
			matrix = None

		if matrix is None:
			matrix = np.eye(shape)

		return matrix.copy()

	@staticmethod
	def permutation_rown(origin_rown_idx: int, dest_rown_idx: int, matrix=None, shape=None, history_matrix=False) -> np.ndarray:

		matrix = History.process_matrix(history_matrix, matrix, shape)

		matrix[origin_rown_idx, :], matrix[dest_rown_idx, :] = matrix[dest_rown_idx, :].copy(), matrix[origin_rown_idx,
																								:].copy()

		return matrix

	@staticmethod
	def permutation_col(origin_rown_idx: int, dest_rown_idx: int, idx_arr: np.ndarray | None, matrix=None, shape=None, history_matrix=False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:

		matrix = History.process_matrix(history_matrix, matrix, shape)

		matrix[:, origin_rown_idx], matrix[:, dest_rown_idx] = matrix[:, dest_rown_idx].copy(), matrix[:,
																								origin_rown_idx].copy()

		if history_matrix:
			return matrix

		if idx_arr is None:
			err_txt = f"""Index array must by exists"""
			raise NameError(err_txt)

		idx_arr = idx_arr.copy()
		idx_arr[origin_rown_idx], idx_arr[dest_rown_idx] = idx_arr[dest_rown_idx], idx_arr[origin_rown_idx]
		return matrix, idx_arr

	@staticmethod
	def multiply_and_subtract_rown(origin_rown_idx: int, dest_rown_idx: int, matrix=None, shape=None, coef=1,
								   history_matrix=False):

		matrix = History.process_matrix(history_matrix, matrix, shape)

		origin_rown = matrix[origin_rown_idx, :]
		dest_rown = matrix[dest_rown_idx, :]

		matrix[dest_rown_idx, :] = dest_rown - (coef * origin_rown)

		return matrix

	@staticmethod
	def divide_line(origin_rown_idx: int, matrix=None, shape=None, coef=1, history_matrix=False):

		matrix = History.process_matrix(history_matrix, matrix, shape)

		origin_rown = matrix[origin_rown_idx, :]

		matrix[origin_rown_idx, :] = origin_rown / coef

		return matrix
