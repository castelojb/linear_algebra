from io import StringIO

import pandas as pd
import streamlit as st

from src.linear_algebra_resolution.gauss_jordam_elimination import gaussian_jordam_elimination
from src.linear_algebra_resolution.gaussian_elimination import gaussian_elimination


def about_page():

	st.text('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')


def linear_algebra_resolution_page():

	uploaded_file = st.file_uploader("Escolha 1 arquivo CSV")

	if uploaded_file is not None:

		x_b = pd.read_csv(uploaded_file).to_numpy()

		matrix = x_b[:, :-2]
		b = x_b[:, -1]

		st.markdown('Equações')
		st.write(matrix)

		st.markdown('Contantes')
		st.write(b)

		options = [
			('Gauss', gaussian_elimination),
			('Gauss-Jordam', gaussian_jordam_elimination)
		]

		option = st.radio(
			'Escolha o metodo para resolver o sistema',
			options,
			format_func=lambda x: x[0]
		)

		method = option[1]
		resp, hist = method(matrix, b)

		st.markdown("Vetor de Resposta")
		st.write(resp)

		st.markdown("Matriz @ Vetor de Resposta")
		st.write(matrix @ resp)

		show_hist = st.checkbox('Historico')
		if show_hist:
			st.markdown('Matrizes de Operações')
			for op_matrix in hist.operations_matrix:
				st.write(op_matrix)

			if option[0] != 'Gauss':

				st.markdown('Inversa')
				inv = hist.get_acum_op_matrix()
				st.write(inv)

				st.markdown('Matriz @ Inversa')
				st.write(matrix @ inv)

def resolve_page(option):

	if option == 'Resolução de Sistemas Lineares':
		linear_algebra_resolution_page()
	if option == 'Sobre':
		about_page()


if __name__ == '__main__':

	st.title('Dashboard da Algebra Linear')
	st.subheader('Totalmente gratuito... mas aceito doações <3')

	options = [
		'Resolução de Sistemas Lineares',
		'Sobre'
	]

	with st.sidebar:
		setor = st.radio(
			'Escolha o setor',
			options
		)

	st.subheader(setor)
	resolve_page(setor)
