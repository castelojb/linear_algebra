import numpy as np
import streamlit as st

from src.linear_algebra_resolution.householder_method import householder


def show_householder_page(a_b: np.ndarray):

	st.markdown('Matriz')
	st.write(a_b)

	ab, h, history = householder(a_b, with_history=True)

	st.markdown('Matriz Householder Acumulada Final')
	st.write(h)

	st.markdown('Matriz de Saída Final')
	st.write(ab)

	if st.checkbox('Historico'):

		for idx, (hc, ab_, h_) in enumerate(history):

			st.markdown(f'[{idx}] Matriz de Householder')
			st.write(hc)

			st.markdown(f'[{idx}] Matriz de Similaridade')
			st.write(ab_)

			st.markdown(f'[{idx}] Matriz de Householder Acumulada')
			st.write(h_)
