from typing import Callable

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from src.linear_algebra_resolution.ordinary_least_squares import least_squares
import plotly.graph_objs as go


def rmse(real_arr: np.ndarray, predicted_arr: np.ndarray) -> float:
	diff = (real_arr - predicted_arr) ** 2

	return np.sqrt(diff.mean())


def z_score_normalization(arr: np.ndarray,
						  with_denomalized=False
						  ) -> np.ndarray | tuple[np.array, Callable[[np.ndarray], np.ndarray]]:
	mean = arr.mean()

	std = arr.std()

	out = (arr - mean) / std

	if with_denomalized:
		return out, lambda x: (std * x) + mean

	return out


def add_ones_column(x: np.ndarray) -> np.ndarray:
	return np.column_stack([
		np.ones(x.shape[0]),
		x
	])


def reshape_vector(x: np.ndarray) -> np.ndarray:
	return x.reshape([-1, 1])


def show_ols_page(x_y: np.ndarray):

	# Data Processing
	df = pd.DataFrame(x_y, columns=['x', 'y'])

	normalized_x = z_score_normalization(df['x'].to_numpy())
	normalized_y, denormalized_y = z_score_normalization(df['y'].to_numpy(), with_denomalized=True)

	X_ones = add_ones_column(normalized_x)
	y = reshape_vector(normalized_y)

	st.markdown("Dados")
	fig = px.scatter(x=df['x'], y=df['y'])
	st.plotly_chart(fig)

	# Regularization
	l2_regulazation = st.slider("Selecione o termo de regularização", min_value=0.1, max_value=1.0, step=0.01)

	# Model
	w = least_squares(X_ones, y, l2_regulazation=l2_regulazation)

	st.markdown('Gradientes')
	st.write(w)

	preds = denormalized_y(X_ones @ w)

	st.markdown('Erro RMSE')
	st.write(
		rmse(y, preds)
	)

	st.markdown('Resultado Final')
	fig = px.scatter(x=df['x'], y=df['y'])
	fig.add_trace(
		go.Scatter(x=df['x'], y=preds[:, 0])
	)
	st.plotly_chart(fig)
