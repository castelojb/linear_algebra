import streamlit as st
from numpy import loadtxt

from src.pages.cg_page import show_cg_page
from src.pages.cholesky_page import show_cholesky_page
from src.pages.eig_qr_method_page import show_eig_qr_page
from src.pages.gauss_jordam_page import show_gauss_jordam_page
from src.pages.gauss_page import show_gauss_page
from src.pages.gram_schmidt_page import show_gram_schmidt_page
from src.pages.householder_page import show_householder_page
from src.pages.least_squares import show_ols_page
from src.pages.lu_page import show_lu_page
from src.pages.power_methods_page import show_power_methods_page
from src.pages.qr_decomposition import show_qr_decomposition_page
from src.pages.rref_page import show_rref_page
from src.pages.sor_page import show_sor_page
from src.pages.svd_page import show_svd_page


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
			('Gram-Schmidt', show_gram_schmidt_page),
			('Metodo dos Minimos Quadrados Ordinarios', show_ols_page),
			('RREF', show_rref_page),
			('Metodos da Potencia', show_power_methods_page),
			('Householder', show_householder_page),
			('Decomposição QR', show_qr_decomposition_page),
			('Auto-Valores/Auto-Vetores QR', show_eig_qr_page),
			('SOR', show_sor_page),
			('SVD', show_svd_page),
			('Gradientes Conjulgados', show_cg_page)
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
