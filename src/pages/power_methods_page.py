import numpy as np
import streamlit as st

from src.linear_algebra_resolution.power_methods import inverse_power_method, \
	regular_power_method, \
	shifted_power_method


def show_power_methods_page(a_b: np.ndarray):

	a = a_b[:, :-1]
	b = a_b[:, -1]

	st.markdown('Matriz')
	st.write(a)

	st.markdown('Chute Inicial')
	st.write(b)

	st.markdown("Potencia Regular")
	regular_eigvector, regular_eigvalue, _ = regular_power_method(a, x=b, iterr_max=1000)
	st.write(regular_eigvalue)
	st.write(regular_eigvector)

	st.markdown("Potencia Inversa")
	inverse_eigvector, inverse_eigvalue, _ = inverse_power_method(a, x=b, iterr_max=1000)
	st.write(inverse_eigvalue)
	st.write(inverse_eigvector)

	st.markdown('Potencia com Deslocamento')
	mu = st.slider('Shifted', min_value=1, max_value=10, step=0.5)
	shifted_eigvector, shifted_eigvalue, _ = shifted_power_method(a, x=b, mu=mu, iterr_max=1000)
	st.write(shifted_eigvalue)
	st.write(shifted_eigvector)
