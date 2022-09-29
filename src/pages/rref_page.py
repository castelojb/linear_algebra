import numpy as np
import streamlit as st
from scipy.linalg import null_space

from src.linear_algebra_resolution.rref import reduced_row_echelon_form


def show_rref_page(x: np.ndarray):

	st.markdown('Equações')
	st.write(x)

	rref_form = reduced_row_echelon_form(x)

	st.markdown('Forma RRE')
	st.write(rref_form)

	st.markdown('Posto da Matriz')
	st.write(np.linalg.matrix_rank(rref_form))

	st.markdown('Dimensão do Espaço Nulo')
	st.write(null_space(rref_form).shape)


