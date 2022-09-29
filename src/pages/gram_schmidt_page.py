import numpy as np
import streamlit as st

from src.linear_algebra_resolution.gram_schmidt import gram_schmidt_from_cholesky


def show_gram_schmidt_page(x: np.ndarray):

	st.markdown('Equações')
	st.write(x)

	g, gm = gram_schmidt_from_cholesky(x)

	st.markdown("Matriz G")
	st.write(g)

	st.markdown("Base Ortonormal Transposta")
	st.write(gm.T)

	inv = np.linalg.inv(gm)

	st.markdown('Inversa da Base')
	st.write(inv)

