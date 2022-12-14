import numpy as np
import streamlit as st

from src.linear_algebra_resolution.qr_decomposition import qr_decomposition, qr_method


def show_eig_qr_page(a_b: np.ndarray):

	st.markdown('Matriz')
	st.write(a_b)

	eignvalues, eigvectors = qr_method(a_b)

	st.markdown('Auto-Valores')
	st.write(eignvalues)

	st.markdown('Auto-Vetores')
	st.write(eigvectors)
