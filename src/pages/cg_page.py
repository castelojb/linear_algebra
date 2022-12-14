import numpy as np
import streamlit as st

from src.linear_algebra_resolution.grad_conj import conjugate_gradient


def show_cg_page(a_b: np.ndarray):

	a = a_b[:, :-1]
	b = a_b[:, -1]

	st.markdown('Matriz')
	st.write(a)

	st.markdown('Vetor de Constantes')
	st.write(b)

	resp = conjugate_gradient(a, b)

	st.markdown('Resposta')
	st.write(resp)
