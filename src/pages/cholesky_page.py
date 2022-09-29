import numpy as np
import streamlit as st

from src.linear_algebra_resolution.cholesky_decomposition import cholesky_decomposition


def show_cholesky_page(x: np.ndarray):

	st.markdown('Equações')
	st.write(x)

	g = cholesky_decomposition(x)

	st.markdown("Matriz G")
	st.write(g)

	st.markdown("G @ G.T")
	st.write(g @ g.T)
