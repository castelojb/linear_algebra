import numpy as np
import streamlit as st

from src.linear_algebra_resolution.qr_decomposition import qr_decomposition


def show_qr_decomposition_page(a_b: np.ndarray):

	st.markdown('Matriz')
	st.write(a_b)

	q, r = qr_decomposition(a_b)

	st.markdown('Matriz Q')
	st.write(q)

	st.markdown('Matriz R')
	st.write(r)

	st.markdown('Produto Q @ R')
	st.write(q @ r)
