import numpy as np
import streamlit as st

from src.linear_algebra_resolution.gauss_jordam_elimination import gaussian_jordam_elimination


def show_gauss_jordam_page(x_b: np.ndarray):

	x = x_b[:, :-1]
	b = x_b[:, -1]

	st.markdown('Equações')
	st.write(x)

	st.markdown('Contantes')
	st.write(b)

	resp, hist = gaussian_jordam_elimination(x, b)

	st.markdown("Vetor de Resposta")
	st.write(resp)

	st.markdown("Matriz @ Vetor de Resposta")
	st.write(x @ resp)

	show_hist = st.checkbox('Historico')
	if show_hist:
		st.markdown('Matrizes de Operações')
		for op_matrix in hist.operations_matrix:
			st.write(op_matrix)

	show_inv = st.checkbox('Inversa')

	if show_inv:
		st.markdown('Inversa')
		inv = hist.get_acum_op_matrix()
		st.write(inv)

		st.markdown('Matriz @ Inversa')
		st.write(x @ inv)
