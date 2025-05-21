import yfinance as yf
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import streamlit as st

@st.cache_data(show_spinner=False)
def get_fama_french_factors():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        # Assume only one file in the zip
        csv_filename = z.namelist()[0]
        with z.open(csv_filename) as f:
            # Skip first 4 lines (metadata), header is line 5
            df = pd.read_csv(f, skiprows=4)
    # Remove footer metadata and empty lines
    df = df.dropna(how='all')
    # Remove any rows where the first column is not a date (should be digits)
    df = df[df.iloc[:,0].astype(str).str.isdigit()]
    # Rename columns for clarity
    df.columns = [col.strip() for col in df.columns]
    # Parse date column
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df.set_index("Date", inplace=True)
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(show_spinner=False)
def get_ticker_histories(tickers, price_type="Close"):
    # Elimina duplicados manteniendo el orden
    histories = []
    failed_tickers = []
    valid_tickers = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, progress=False, auto_adjust=True)
            if not data.empty and price_type in data.columns:
                selected = data[[price_type]]
                selected.columns = [ticker]
                histories.append(selected)
                valid_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
        except Exception as e:
            failed_tickers.append(ticker)
    
    try:
        # Obtener y unir los factores de Fama-French
        ff_factors = get_fama_french_factors()
    except Exception as e:
        st.error("Error al obtener los factores de Fama-French. Asegúrate de que la URL sea accesible.")
        ff_factors = pd.DataFrame()
    
    if histories:
        df = pd.concat(histories, axis=1, join='outer')
        df = df.dropna(how='all')  # Elimina filas donde todos los valores son NaN
        df.columns.name = None
        # Calcular retornos: ln(p(n)/p(n-1)) * 100
        returns = np.log(df / df.shift(1)) * 100
        returns = returns.dropna(how='all')

        # Unir por fecha (índice)
        combined = returns.join(ff_factors, how='left')
        return combined, failed_tickers, valid_tickers       
    else:
        return ff_factors, tickers, []  # Todos fallaron