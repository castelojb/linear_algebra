import numpy as np
import streamlit as st

from src.linear_algebra_resolution.svd_decomposition import svd


def show_svd_page(a_b: np.ndarray):

	st.markdown('Matriz')
	st.write(a_b)

	singular_values, us, vs = svd(a_b)

	st.markdown('Matriz U')
	st.write(us)

	st.markdown('Matriz Sigma')
	st.write(singular_values)

	st.markdown('Matriz VT')
	st.write(vs)

	st.markdown('Produto U @ Sigma @ VT')
	st.write(us @ singular_values @ vs)
