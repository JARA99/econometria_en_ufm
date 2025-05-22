import streamlit as st
import pandas as pd
import datetime as dt
from process import get_ticker_histories
from tabs import tab_0, tab_1, tab_2, tab_3

price_type = 'Close'  # Tipo de precio a usar (Close, Open, High, Low)

# Leer tickers desde el archivo CSV
tickers_df = pd.read_csv("https://raw.githubusercontent.com/JARA99/econometria_en_ufm/refs/heads/main/tarea_final/tickers.csv", header=None)
all_tickers = tickers_df[0].tolist()

st.title("Optimización de portafolio")

col1, col2 = st.columns([2, 2])
with col1:
    date_range = st.date_input(
        "Seleccione el rango de fechas para el **análisis**:",
        value=(dt.date(2021, 7, 1), dt.date(2024, 6, 30))
    )
with col2:
    sim_date_range = st.date_input(
        "Seleccione el rango de fechas para la **simulación**:",
        value=(dt.date(2024, 7, 1), dt.date(2024, 12, 31))
    )

tickers = st.multiselect(
    "Seleccione los tickers de las empresas a optimizar:",
    options=all_tickers,
    placeholder="Ej: AAPL, AMZN, ...",
    accept_new_options=True,
    default=['GLD', 'BTC-USD', 'BRK-B', 'SPY', 'IRBO']
)

returns_df, prices, failed_tickers, valid_tickers = get_ticker_histories(tickers, price_type)

returns_df = returns_df.dropna()
prices = prices.dropna()

if not returns_df.empty and len(returns_df.columns) == len(set(returns_df.columns)):
    # Limitar la visualización al rango seleccionado
    start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    sim_start, sim_end = pd.to_datetime(sim_date_range[0]), pd.to_datetime(sim_date_range[1])
    returns_df_analysis = returns_df[(returns_df.index >= start) & (returns_df.index <= end)]
    if not prices.empty:
        prices_analysis = prices[(prices.index >= start) & (prices.index <= end)]
    else:
        prices_analysis = pd.DataFrame()
    # Filter returns for simulation period
    sim_returns = returns_df[(returns_df.index >= sim_start) & (returns_df.index <= sim_end)]

    with st.expander("Ver datos de retornos", expanded=False):
        st.dataframe(returns_df)
        if failed_tickers:
            st.warning(f"Tickers no encontrados o sin datos: {', '.join(failed_tickers)}")

    # Compute excess returns
    e_returns = returns_df_analysis[valid_tickers].copy()
    e_returns = e_returns.sub(returns_df_analysis['RF'], axis=0)

    # Add global USD amount input before tabs
    usd_amount = st.number_input("Monto total a invertir (USD)", min_value=1.0, value=5000.0, step=100.0, key="usd_amount_global")

    # Add global yearly toggle before tabs
    ff_capm = st.toggle("Usar Fama-French en lugar de CAPM", value=False, key="toggle_FF_CAPM")

    if ff_capm:
        compare_cols = ['Mkt-RF', 'SMB', 'HML']
    else:
        compare_cols = ['Mkt-RF']

    # Add global yearly toggle before tabs
    yearly = st.toggle("Mostrar valores anualizados", value=False, key="toggle_yearly_global")

    # Tabs para análisis
    tab0, tab1, tab2, tab3 = st.tabs([
        "Configuración inicial",
        "Minimizar varianza",
        "Maximizar sharpe",
        "Maximizar Alpha"
    ])
    with tab0:
        compare_vals = tab_0(
            e_returns, prices_analysis, yearly=yearly, usd_amount=usd_amount,
            sim_returns=sim_returns, compare_returns=returns_df_analysis[compare_cols], compare_cols=compare_cols
        )
    with tab1:
        tab_1(
            e_returns, compare_vals, prices_analysis, yearly=yearly, usd_amount=usd_amount,
            sim_returns=sim_returns, compare_returns=returns_df_analysis[compare_cols], compare_cols=compare_cols
        )
    with tab2:
        tab_2(
            e_returns, compare_vals, prices_analysis, yearly=yearly, usd_amount=usd_amount,
            sim_returns=sim_returns, compare_returns=returns_df_analysis[compare_cols], compare_cols=compare_cols
        )
    with tab3:
        tab_3(
            e_returns, compare_vals, prices_analysis, yearly=yearly, usd_amount=usd_amount,
            sim_returns=sim_returns, compare_returns=returns_df_analysis[compare_cols], compare_cols=compare_cols
        )
else:
    st.rerun()
