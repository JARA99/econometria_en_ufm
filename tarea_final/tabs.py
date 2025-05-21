import streamlit as st
import pandas as pd
from plotly import express as px
import numpy as np
import plotly.graph_objects as go

def tab_0(valid_tickers, returns_df_analysis):
    n = len(valid_tickers)
    weights = [1/n] * n
    w_table(valid_tickers, weights)
    data_weighted, port_mean, port_std = get_weighted_portfolio_data(returns_df_analysis, weights, valid_tickers)
    shewheart_plot(data_weighted, port_mean, port_std)

def w_table(tickers, weights):
    # Verificar si hay tickers válidos
    if not tickers:
        st.info("No hay tickers válidos para mostrar pesos iniciales.")
        return
    # Crear encabezados
    headers = [f"W_{ticker}" for ticker in tickers]
    # Crear DataFrame para mostrar
    df_weights = pd.DataFrame([weights], columns=headers)
    # Mostrar DataFrame
    st.dataframe(df_weights, hide_index=True)

def get_weighted_portfolio_data(data, weights, tickers):
    # Calcula los retornos ponderados, media y desviación estándar
    data_weighted = data[tickers].copy()
    for ticker, weight in zip(tickers, weights):
        data_weighted[ticker] = data_weighted[ticker] * weight
    data_weighted["Portfolio"] = data_weighted.sum(axis=1)
    port_mean = data_weighted["Portfolio"].mean()
    port_std = data_weighted["Portfolio"].std(ddof=1)
    return data_weighted, port_mean, port_std

def shewheart_plot(data_weighted, port_mean, port_std):
    # Portfolio plot with mean and std bands
    fig = go.Figure()

    # Portfolio line (olive)
    fig.add_trace(go.Scatter(
        x=data_weighted.index,
        y=data_weighted["Portfolio"],
        mode="lines",
        name="Portfolio",
        line=dict(color="olive", width=2)
    ))

    # Mean line (μ, orangered)
    fig.add_trace(go.Scatter(
        x=data_weighted.index,
        y=[port_mean]*len(data_weighted),
        mode="lines",
        name="μ",
        line=dict(color="orangered", dash="dash")
    ))
    # Std lines (μ ± σ, orangered)
    fig.add_trace(go.Scatter(
        x=data_weighted.index,
        y=[port_mean + port_std]*len(data_weighted),
        mode="lines",
        name="σ",
        line=dict(color="orangered", dash="dot", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=data_weighted.index,
        y=[port_mean - port_std]*len(data_weighted),
        mode="lines",
        name="σ",
        line=dict(color="orangered", dash="dot", width=1.5),
        showlegend=False
    ))
    # 3*Std lines (μ ± 3σ, orangered)
    fig.add_trace(go.Scatter(
        x=data_weighted.index,
        y=[port_mean + 3*port_std]*len(data_weighted),
        mode="lines",
        name="3σ",
        line=dict(color="orangered", dash="dashdot", width=1)
    ))
    fig.add_trace(go.Scatter(
        x=data_weighted.index,
        y=[port_mean - 3*port_std]*len(data_weighted),
        mode="lines",
        name="3σ",
        line=dict(color="orangered", dash="dashdot", width=1),
        showlegend=False
    ))

    fig.update_layout(
        title="Retorno del portafolio con bandas de media y desviación estándar",
        yaxis_title="Retorno",
        xaxis_title="Fecha",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1a", step="year", stepmode="backward"),
                    dict(label="Todo", step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date",
        )
    )

    st.plotly_chart(fig, use_container_width=True)
