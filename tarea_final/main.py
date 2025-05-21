import streamlit as st
import pandas as pd
import datetime as dt
from process import get_ticker_histories
from tabs import tab_0


# Leer tickers desde el archivo CSV
tickers_df = pd.read_csv("tickers.csv", header=None)
all_tickers = tickers_df[0].tolist()

st.title("Optimización de portafolio")

col1, col2 = st.columns([2, 2])
with col1:
    price_type = st.segmented_control(
        "Seleccione el tipo de precio a analizar:",
        options=["Open", "High", "Low", "Close"],
        selection_mode="single",
        default="Close",
    )
with col2:
    date_range = st.date_input(
        "Seleccione el rango de fechas para el análisis:",
        value=(dt.date(2022, 1, 1), dt.date(2024, 12, 31))
    )

tickers = st.multiselect(
    "Seleccione los tickers de las empresas a optimizar:",
    options=all_tickers,
    placeholder="Ej: AAPL, AMZN, ...",
    accept_new_options=True,
)

returns_df, failed_tickers, valid_tickers = get_ticker_histories(tickers, price_type)

if not returns_df.empty and len(returns_df.columns) == len(set(returns_df.columns)):
    # Limitar la visualización al rango seleccionado
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    returns_df_analysis = returns_df[(returns_df.index >= start) & (returns_df.index <= end)]
    # print(returns_df_analysis)
    # print(tickers)
    with st.expander("Ver datos de retornos", expanded=False):
        st.dataframe(returns_df)
        if failed_tickers:
            st.warning(f"Tickers no encontrados o sin datos: {', '.join(failed_tickers)}")

    # Tabs para análisis
    tab0, tab1, tab2, tab3 = st.tabs([
        "Configuración inicial",
        "Minimizar varianza",
        "Maximizar sharpe",
        "Maximizar Alpha"
    ])
    with tab0:
        tab_0(valid_tickers, returns_df_analysis)
    with tab1:
        st.write("Análisis para minimizar varianza (pendiente de implementación).")
    with tab2:
        st.write("Análisis para maximizar Sharpe (pendiente de implementación).")
    with tab3:
        st.write("Análisis para maximizar Alpha (pendiente de implementación).")
else:
    st.rerun()
