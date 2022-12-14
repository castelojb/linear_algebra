import numpy as np
import streamlit as st

from src.linear_algebra_resolution.sor import sor_solver


def show_sor_page(a_b: np.ndarray):

	a = a_b[:, :-1]
	b = a_b[:, -1]

	st.markdown('Matriz')
	st.write(a)

	st.markdown('Vetor de Constantes')
	st.write(b)

	# omega = st.slider('Omega', min_value=1.5, max_value=10.5, step=0.5)

	resp = sor_solver(a, b, 1.5)

	st.markdown('Resposta')
	st.write(resp)
