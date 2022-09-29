import streamlit as st
from numpy import loadtxt

from src.pages.cholesky_page import show_cholesky_page
from src.pages.gauss_jordam_page import show_gauss_jordam_page
from src.pages.gauss_page import show_gauss_page
from src.pages.gram_schmidt_page import show_gram_schmidt_page
from src.pages.lu_page import show_lu_page


def about_page():

	st.markdown('Desenvolvido por João Araújo Castelo Branco para fins didaticos')
	st.markdown('Email: joaocb14@gmail.com')


def linear_algebra_resolution_page():

	uploaded_file = st.file_uploader("Escolha 1 arquivo CSV", type=['csv'])

	if uploaded_file is not None:

		x = loadtxt(uploaded_file, delimiter=',')

		page_options = [
			('Gauss', show_gauss_page),
			('Gauss-Jordam', show_gauss_jordam_page),
			('Lower-Upper', show_lu_page),
			('Cholesk Decomposition', show_cholesky_page),
			('Gram-Schmidt', show_gram_schmidt_page)
		]

		page_option = st.radio(
			'Escolha o metodo para resolver o sistema',
			page_options,
			format_func=lambda tuple_: tuple_[0]
		)

		page = page_option[1]
		page(x)


if __name__ == '__main__':

	st.title('Dashboard da Algebra Linear')
	st.subheader('Totalmente gratuito... mas aceito doações <3')

	options = [
		('Sobre', about_page),
		('Resolução de Sistemas Lineares', linear_algebra_resolution_page),
	]

	with st.sidebar:
		option = st.radio(
			'Escolha o setor',
			options,
			format_func=lambda x: x[0]
		)

	st.subheader(option[0])
	option[1]()
