import numpy as np
import streamlit as st

from src.linear_algebra_resolution.lu_decomposition import lu_resolution


def show_lu_page(x_b: np.ndarray):

	x = x_b[:, :-1]
	b = x_b[:, -1]

	st.markdown('Equações')
	st.write(x)

	st.markdown('Contantes')
	st.write(b)

	l, u, resp = lu_resolution(x, b)

	st.markdown("Vetor de Resposta")
	st.write(resp)

	st.markdown("Matriz @ Vetor de Resposta")
	st.write(x @ resp)

	st.markdown('Lower Matrix')
	st.write(l)

	st.markdown('Upper Matrix')
	st.write(u)

	st.markdown('Lower @ Upper')
	st.write(l @ u)
