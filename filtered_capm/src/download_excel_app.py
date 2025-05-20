import streamlit as st
import yfinance as yf
import pandas as pd
from io import BytesIO

st.title("Descargar datos de Yahoo Finance a Excel")

tickers = st.multiselect(
    "Selecciona los tickers:",
    ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA", "SPY", "DIA", "QQQ", "IWM"]
)

if tickers:
    st.write(f"Tickers seleccionados: {', '.join(tickers)}")
    data = yf.download(tickers, period="max", group_by='ticker', auto_adjust=True)
    # If only one ticker, make it a DataFrame with one level columns for consistency
    if len(tickers) == 1:
        data.columns = pd.MultiIndex.from_product([tickers, data.columns])
    # Flatten columns for Excel
    data.columns = ['_'.join(col).strip() for col in data.columns.values]
    # Download button
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, sheet_name="Datos")
    st.download_button(
        label="Descargar Excel",
        data=output.getvalue(),
        file_name="datos_yahoo_finance.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
else:
    st.info("Selecciona al menos un ticker para descargar los datos.")
