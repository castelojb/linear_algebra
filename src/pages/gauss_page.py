import numpy as np
import streamlit as st

from src.linear_algebra_resolution.gaussian_elimination import gaussian_elimination


def show_gauss_page(x_b: np.ndarray):

	x = x_b[:, :-1]
	b = x_b[:, -1]

	st.markdown('Equações')
	st.write(x)

	st.markdown('Contantes')
	st.write(b)

	resp, hist, x_forward = gaussian_elimination(x, b)

	st.markdown("Vetor de Resposta")
	st.write(resp)

	st.markdown("Matriz @ Vetor de Resposta")
	st.write(x @ resp)

	show_hist = st.checkbox('Historico')

	if show_hist:

		st.markdown('Matriz Final')
		st.write(x_forward)

		st.markdown('Matrizes de Operações')
		for op_matrix in hist.operations_matrix:
			st.write(op_matrix)
