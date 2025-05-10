import streamlit as st
import yfinance as yf
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

def adjust_returns_within_limits(returns, mean, std, std_multiplier):
    """
    Adjust return values within the limits by pushing them closer to the mean.
    The adjustment is stronger near the mean and softer as it approaches the limits.
    Values outside the limits remain unchanged.

    :param returns: Series of return values
    :param mean: Mean of the returns
    :param std: Standard deviation of the returns
    :param std_multiplier: Number of standard deviations defining the limits
    :return: Adjusted return values
    """
    lower_limit = mean - std_multiplier * std
    upper_limit = mean + std_multiplier * std

    def adjust_value(value):
        if lower_limit <= value <= upper_limit:
            # Calculate the adjustment factor based on proximity to the mean
            proximity = 1 - abs(value - mean) / (std_multiplier * std)
            return mean + (value - mean) * proximity
        return value  # Return unchanged if outside the limits

    return returns.apply(adjust_value)

def calculate_fit(x, y):
    """
    Calculate the linear regression fit for two variables.

    :param x: Independent variable (reference ticker returns)
    :param y: Dependent variable (selected ticker returns)
    :return: slope, intercept, r_squared
    """
    model = LinearRegression()
    x_reshaped = x.values.reshape(-1, 1)
    model.fit(x_reshaped, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(x_reshaped, y)
    return slope, intercept, r_squared

def main():
    st.title("CAPM con filtro de ruido")
    st.write("¡Bienvenidos!")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuración")
        reference_ticker = st.selectbox(
            "Selecciona un activo de referencia:",
            ["SPY", "DIA", "QQQ", "IWM"],
            index=0  # Default to 'SPY'
        )
        st.sidebar.markdown(f"### Activo de referencia seleccionado: {reference_ticker}")
        std_multiplier = st.slider(
            "Selecciona el número de desviaciones estándar:",
            min_value=0.0,
            max_value=10.0,
            value=3.0,  # Default to 3
            step=0.5,
            format="%.1f"
        )

    # List of common tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NFLX", "NVDA"]

    # Selectbox for ticker selection
    selected_ticker = st.selectbox(
        "Selecciona un activo:",
        tickers,
        accept_new_options=True,
    )

    # Display selected ticker
    st.markdown(f"## {selected_ticker}")

    # Fetch data for the selected ticker and reference ticker
    if selected_ticker and reference_ticker:
        ticker_data = yf.Ticker(selected_ticker)
        reference_data = yf.Ticker(reference_ticker)

        historical_data = ticker_data.history(period="max")
        reference_historical_data = reference_data.history(period="max")

        # Ensure both datasets are not empty
        if not historical_data.empty and not reference_historical_data.empty:
            # Find the earliest common date
            common_start_date = max(historical_data.index.min(), reference_historical_data.index.min())

            # Filter data to start from the common date
            historical_data = historical_data[historical_data.index >= common_start_date]
            reference_historical_data = reference_historical_data[reference_historical_data.index >= common_start_date]

            # Add a new column for normalized close values
            historical_data["Normalized Close"] = historical_data["Close"] / historical_data["Close"].iloc[0] * 100
            reference_historical_data["Normalized Close"] = reference_historical_data["Close"] / reference_historical_data["Close"].iloc[0] * 100

            # Combine both datasets for plotting normalized close
            combined_data = historical_data[["Normalized Close"]].rename(columns={"Normalized Close": selected_ticker}).join(
                reference_historical_data[["Normalized Close"]].rename(columns={"Normalized Close": reference_ticker}),
                how="inner"
            )

            # Plot normalized close
            fig = px.line(
                combined_data,
                x=combined_data.index,
                y=combined_data.columns,
                title=f"Crecimiento porcentual: {selected_ticker} y {reference_ticker}",
                labels={"Date": "Fecha", "value": "Crecimiento (%)", "variable": "Activo"}
            )
            st.plotly_chart(fig)

            # Calculate returns for both assets
            historical_data["Returns"] = np.log(historical_data["Close"] / historical_data["Close"].shift(1))
            reference_historical_data["Returns"] = np.log(reference_historical_data["Close"] / reference_historical_data["Close"].shift(1))

            # Combine both datasets for plotting returns
            returns_data = historical_data[["Returns"]].rename(columns={"Returns": selected_ticker}).join(
                reference_historical_data[["Returns"]].rename(columns={"Returns": reference_ticker}),
                how="inner"
            ).dropna()  # Drop NaN values caused by the shift operation

            # Calculate mean and standard deviation for each asset
            mean_selected = returns_data[selected_ticker].mean()
            std_selected = returns_data[selected_ticker].std()
            mean_reference = returns_data[reference_ticker].mean()
            std_reference = returns_data[reference_ticker].std()

            # Plot returns with horizontal lines for mean and ± selected standard deviations
            fig_returns = px.line(
                returns_data,
                x=returns_data.index,
                y=returns_data.columns,
                title=f"Rendimientos diarios: {selected_ticker} y {reference_ticker}",
                labels={"Date": "Fecha", "value": "Rendimientos", "variable": "Activo"}
            )

            # Add horizontal lines for selected ticker
            fig_returns.add_hline(y=mean_selected, line_dash="dash", line_color="blue", annotation_text=f"Media {selected_ticker}")
            fig_returns.add_hline(y=mean_selected + std_multiplier * std_selected, line_dash="dot", line_color="blue", annotation_text=f"+{std_multiplier}σ {selected_ticker}")
            fig_returns.add_hline(y=mean_selected - std_multiplier * std_selected, line_dash="dot", line_color="blue", annotation_text=f"-{std_multiplier}σ {selected_ticker}")

            # Add horizontal lines for reference ticker
            fig_returns.add_hline(y=mean_reference, line_dash="dash", line_color="red", annotation_text=f"Media {reference_ticker}")
            fig_returns.add_hline(y=mean_reference + std_multiplier * std_reference, line_dash="dot", line_color="red", annotation_text=f"+{std_multiplier}σ {reference_ticker}")
            fig_returns.add_hline(y=mean_reference - std_multiplier * std_reference, line_dash="dot", line_color="red", annotation_text=f"-{std_multiplier}σ {reference_ticker}")

            st.plotly_chart(fig_returns)

            # Display mean and standard deviation under the returns figure
            st.markdown(f"**Estadísticas de los rendimientos:**")
            st.markdown(f"- **{selected_ticker}:** Media = {mean_selected:.4f}, Sigma (σ) = {std_selected:.4f}")
            st.markdown(f"- **{reference_ticker}:** Media = {mean_reference:.4f}, Sigma (σ) = {std_reference:.4f}")

            # Adjust returns within the limits
            adjusted_returns_selected = adjust_returns_within_limits(
                returns_data[selected_ticker], mean_selected, std_selected, std_multiplier
            )
            adjusted_returns_reference = adjust_returns_within_limits(
                returns_data[reference_ticker], mean_reference, std_reference, std_multiplier
            )

            # Combine adjusted returns for plotting
            adjusted_returns_data = adjusted_returns_selected.rename(selected_ticker).to_frame().join(
                adjusted_returns_reference.rename(reference_ticker).to_frame()
            )

            # Plot adjusted returns
            fig_adjusted_returns = px.line(
                adjusted_returns_data,
                x=adjusted_returns_data.index,
                y=adjusted_returns_data.columns,
                title=f"Rendimientos ajustados: {selected_ticker} y {reference_ticker}",
                labels={"Date": "Fecha", "value": "Rendimientos ajustados", "variable": "Activo"}
            )
            st.plotly_chart(fig_adjusted_returns)

            # Scatter plot for original returns
            slope_original, intercept_original, r_squared_original = calculate_fit(
                returns_data[reference_ticker], returns_data[selected_ticker]
            )
            fig_scatter_original = px.scatter(
                returns_data,
                x=reference_ticker,
                y=selected_ticker,
                title=f"Dispersión de rendimientos originales: {reference_ticker} vs {selected_ticker}",
                labels={reference_ticker: f"Rendimientos {reference_ticker}", selected_ticker: f"Rendimientos {selected_ticker}"}
            )
            fig_scatter_original.add_scatter(
                x=returns_data[reference_ticker],
                y=slope_original * returns_data[reference_ticker] + intercept_original,
                mode="lines",
                name="Ajuste lineal",
                line=dict(color="red")
            )
            st.plotly_chart(fig_scatter_original)
            st.markdown(f"**Estadísticas del ajuste (original):**\n- Pendiente: {slope_original:.4f}\n- Intercepto: {intercept_original:.4f}\n- R²: {r_squared_original:.4f}")

            # Scatter plot for adjusted returns
            slope_adjusted, intercept_adjusted, r_squared_adjusted = calculate_fit(
                adjusted_returns_data[reference_ticker], adjusted_returns_data[selected_ticker]
            )
            fig_scatter_adjusted = px.scatter(
                adjusted_returns_data,
                x=reference_ticker,
                y=selected_ticker,
                title=f"Dispersión de rendimientos ajustados: {reference_ticker} vs {selected_ticker}",
                labels={reference_ticker: f"Rendimientos ajustados {reference_ticker}", selected_ticker: f"Rendimientos ajustados {selected_ticker}"}
            )
            fig_scatter_adjusted.add_scatter(
                x=adjusted_returns_data[reference_ticker],
                y=slope_adjusted * adjusted_returns_data[reference_ticker] + intercept_adjusted,
                mode="lines",
                name="Ajuste lineal",
                line=dict(color="blue")
            )
            st.plotly_chart(fig_scatter_adjusted)
            st.markdown(f"**Estadísticas del ajuste (ajustado):**\n- Pendiente: {slope_adjusted:.4f}\n- Intercepto: {intercept_adjusted:.4f}\n- R²: {r_squared_adjusted:.4f}")

            # Scatter plot for returns within sigma range for both tickers
            mask_selected = (
                (returns_data[selected_ticker] >= mean_selected - std_multiplier * std_selected) &
                (returns_data[selected_ticker] <= mean_selected + std_multiplier * std_selected)
            )
            mask_reference = (
                (returns_data[reference_ticker] >= mean_reference - std_multiplier * std_reference) &
                (returns_data[reference_ticker] <= mean_reference + std_multiplier * std_reference)
            )
            mask_both = mask_selected & mask_reference
            returns_within_sigma = returns_data[mask_both]

            if not returns_within_sigma.empty:
                slope_sigma, intercept_sigma, r_squared_sigma = calculate_fit(
                    returns_within_sigma[reference_ticker], returns_within_sigma[selected_ticker]
                )
                fig_scatter_sigma = px.scatter(
                    returns_within_sigma,
                    x=reference_ticker,
                    y=selected_ticker,
                    title=f"Dispersión dentro de ±{std_multiplier}σ: {reference_ticker} vs {selected_ticker}",
                    labels={reference_ticker: f"Rendimientos {reference_ticker}", selected_ticker: f"Rendimientos {selected_ticker}"}
                )
                fig_scatter_sigma.add_scatter(
                    x=returns_within_sigma[reference_ticker],
                    y=slope_sigma * returns_within_sigma[reference_ticker] + intercept_sigma,
                    mode="lines",
                    name="Ajuste lineal",
                    line=dict(color="green")
                )
                st.plotly_chart(fig_scatter_sigma)
                st.markdown(
                    f"**Estadísticas del ajuste (dentro de ±{std_multiplier}σ):**\n"
                    f"- Pendiente: {slope_sigma:.4f}\n"
                    f"- Intercepto: {intercept_sigma:.4f}\n"
                    f"- R²: {r_squared_sigma:.4f}"
                )

            # Scatter plot for returns outside sigma range for both tickers
            mask_selected_out = (
                (returns_data[selected_ticker] < mean_selected - std_multiplier * std_selected) |
                (returns_data[selected_ticker] > mean_selected + std_multiplier * std_selected)
            )
            mask_reference_out = (
                (returns_data[reference_ticker] < mean_reference - std_multiplier * std_reference) |
                (returns_data[reference_ticker] > mean_reference + std_multiplier * std_reference)
            )
            mask_both_out = mask_selected_out & mask_reference_out
            returns_outside_sigma = returns_data[mask_both_out]

            if not returns_outside_sigma.empty:
                slope_sigma_out, intercept_sigma_out, r_squared_sigma_out = calculate_fit(
                    returns_outside_sigma[reference_ticker], returns_outside_sigma[selected_ticker]
                )
                fig_scatter_sigma_out = px.scatter(
                    returns_outside_sigma,
                    x=reference_ticker,
                    y=selected_ticker,
                    title=f"Dispersión fuera de ±{std_multiplier}σ: {reference_ticker} vs {selected_ticker}",
                    labels={reference_ticker: f"Rendimientos {reference_ticker}", selected_ticker: f"Rendimientos {selected_ticker}"}
                )
                fig_scatter_sigma_out.add_scatter(
                    x=returns_outside_sigma[reference_ticker],
                    y=slope_sigma_out * returns_outside_sigma[reference_ticker] + intercept_sigma_out,
                    mode="lines",
                    name="Ajuste lineal",
                    line=dict(color="orange")
                )
                st.plotly_chart(fig_scatter_sigma_out)
                st.markdown(
                    f"**Estadísticas del ajuste (fuera de ±{std_multiplier}σ):**\n"
                    f"- Pendiente: {slope_sigma_out:.4f}\n"
                    f"- Intercepto: {intercept_sigma_out:.4f}\n"
                    f"- R²: {r_squared_sigma_out:.4f}"
                )

if __name__ == "__main__":
    main()