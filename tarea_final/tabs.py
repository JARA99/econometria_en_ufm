import streamlit as st
import pandas as pd
from plotly import express as px
import numpy as np
import plotly.graph_objects as go
from process import get_mean_and_std
from scipy.optimize import minimize
from scipy import stats

def w_table(tickers, weights):
    # Crear encabezados
    headers = [f"W_{ticker}" for ticker in tickers]
    # Crear DataFrame para mostrar
    df_weights = pd.DataFrame([weights], columns=headers)
    # Mostrar DataFrame
    st.dataframe(df_weights, hide_index=True)

def show_portfolio_results(
    e_returns, weights, p_mean, p_std, p_sharpe,
    prices=None, compare_vals=None, yearly=False, usd_amount=5000.0, sim_returns=None, alpha=None, betas=None,
    compare_cols=['Mkt-RF', 'SMB', 'HML'], sim_plot_key=None
):
    # Use yearly flag from argument
    if yearly:
        p_mean = p_mean * 252
        p_std = p_std * np.sqrt(252)
        if compare_vals is not None:
            compare_vals = (compare_vals[0] * 252, compare_vals[1] * np.sqrt(252), compare_vals[2], compare_vals[3])
    # Display results as st.metrics, optionally with deltas
    col1, col2, col3, col4 = st.columns(4)
    if compare_vals is not None:
        with col1:
            st.metric(
                label="Media del portafolio",
                value=f"{p_mean:.2f} %",
                delta=f"{((p_mean - compare_vals[0])/abs(compare_vals[0])):.2f} %"
            )
        with col2:
            st.metric(
                label="Desviación estándar del portafolio",
                value=f"{p_std:.2f} %",
                delta=f"{((p_std - compare_vals[1])/abs(compare_vals[1])):.2f} %",
                delta_color="inverse"
            )
        with col3:
            st.metric(
                label="Sharpe",
                value=f"{p_sharpe:.2f}",
                delta=f"{((p_sharpe - compare_vals[2])/abs(compare_vals[2])):.2f} %"
            )
        with col4:
            st.metric(
                label="Alpha de Jensen",
                value=f"{alpha:.4f}" if alpha is not None else "-",
                delta=f"{((alpha - compare_vals[3])/abs(compare_vals[3])):.2f} %"
            )
    else:
        with col1:
            st.metric(label="Media del portafolio", value=f"{p_mean:.2f} %")
        with col2:
            st.metric(label="Desviación estándar del portafolio", value=f"{p_std:.2f} %")
        with col3:
            st.metric(label="Sharpe", value=f"{p_sharpe:.2f}")
        with col4:
            st.metric(label="Alpha de Jensen", value=f"{alpha:.4f}" if alpha is not None else "-")

    # --- Investment allocation table using provided prices ---
    if prices is not None and not prices.empty:
        st.subheader("Asignación de acciones por ticker")
        last_date = prices.index.max()
        alloc_data = []
        for ticker, weight in zip(e_returns.columns, weights):
            # Get last available price for ticker
            price = prices[ticker].loc[last_date] if ticker in prices.columns and last_date in prices.index else np.nan
            usd_alloc = usd_amount * weight
            n_shares = np.floor(usd_alloc / price) if isinstance(price, (float, int, np.floating, np.integer)) and price > 0 else np.nan
            actual_spent = n_shares * price if not np.isnan(n_shares) and not np.isnan(price) else np.nan
            alloc_data.append({
                "Ticker": ticker,
                "Peso": weight,
                "Precio (USD)": f"${price:,.2f}" if not np.isnan(price) else "-",
                "USD asignados": f"${usd_alloc:,.2f}",
                "Acciones a comprar": int(n_shares) if not np.isnan(n_shares) else "-",
                "USD gastados": f"${actual_spent:,.2f}" if not np.isnan(actual_spent) else "-"
            })
        alloc_df = pd.DataFrame(alloc_data)
        st.dataframe(alloc_df, hide_index=True, use_container_width=True)
        # Show betas under the table if available
        if betas is not None and isinstance(betas, dict):
            st.markdown("**Betas estimados:**")
            beta_str = ", ".join([f"{k}: {v:.4f}" for k, v in betas.items()])
            st.info(beta_str)
    
    with st.expander("Mostrar matrices", expanded=False):
        # Show weights
        st.subheader("Pesos del portafolio")
        w_table(e_returns.columns, weights)

        # Show returns vector
        st.subheader("Vector de Retornos")
        st.dataframe(e_returns.mean().to_frame().T, use_container_width=True, hide_index=True)

        # Show cov matrix
        st.subheader("Matriz de Covarianza")
        st.dataframe(e_returns.cov(), use_container_width=True)

    st.header("Simulación")
    # --- Shewhart plot for simulation period ---
    if sim_returns is not None:
        tickers = [t for t in e_returns.columns if t in sim_returns.columns]
        if tickers:
            port_ret = sim_returns[tickers].copy()
            for ticker, weight in zip(tickers, weights):
                port_ret[ticker] = port_ret[ticker] * weight
            port_ret["Portfolio"] = port_ret.sum(axis=1)
            port_mean = port_ret["Portfolio"].mean()
            port_std = port_ret["Portfolio"].std(ddof=1)
            shewheart_plot(port_ret, port_mean, port_std, key=sim_plot_key)

            # --- Calculate real alpha and betas for simulation period ---
            real_alpha, real_betas = get_alpha_beta(port_ret["Portfolio"], sim_returns[compare_cols])

            # --- For each parameter, calculate the probability that the estimated value is correct for the simulated data ---
            df = pd.DataFrame({"portfolio": port_ret["Portfolio"]})
            for col in compare_cols:
                df[col] = sim_returns[col]
            df = df.dropna()  # Drop rows with NaN values
            X = df[compare_cols].values
            y = df["portfolio"].values
            X_mat = np.column_stack([np.ones(len(X)), X])
            n, k = X_mat.shape

            # For each parameter, set the estimated value as the null hypothesis and compute the probability
            prob_results = []
            param_names = ["Alpha"] + compare_cols
            est_vals = [alpha] + [betas[k] for k in compare_cols] if (alpha is not None and betas is not None) else [np.nan] * (1 + len(compare_cols))
            real_vals = [real_alpha] + [real_betas[k] for k in compare_cols]

            for i, (param, est_val) in enumerate(zip(param_names, est_vals)):
                # Build new y under the null hypothesis: parameter = est_val, others = real values
                # For alpha:
                if i == 0:
                    X_h = X_mat.copy()
                    X_h[:, 0] = 1  # intercept
                    beta_h = np.array([est_val] + [real_betas[k] for k in compare_cols])
                else:
                    X_h = X_mat.copy()
                    beta_h = np.array([real_alpha] + [real_betas[k] for k in compare_cols])
                    beta_h[i] = est_val  # set the i-th parameter to the hypothesis value

                y_h = X_h @ beta_h
                residuals_h = y - y_h
                sigma2_h = np.sum(residuals_h**2) / (n - k)
                cov_beta_h = sigma2_h * np.linalg.inv(X_mat.T @ X_mat)
                se_h = np.sqrt(np.diag(cov_beta_h))[i]
                t_stat = (real_vals[i] - est_val) / se_h
                p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - k))
                prob_results.append({
                    "Parámetro": param,
                    "Estimado": f"{est_val:.4f}" if not np.isnan(est_val) else "-",
                    "Simulado": f"{real_vals[i]:.4f}" if not np.isnan(real_vals[i]) else "-",
                    "Probabilidad Hipótesis": f"{1-p_val:.4f}" if not np.isnan(p_val) else "-"
                })

            st.subheader("Valores obtenidos durante la simulación")

            port_sharpe = port_mean / port_std if port_std != 0 else np.nan
            
            sim_cols = st.columns(4)

            if yearly:
                port_mean = port_mean * 252
                port_std = port_std * np.sqrt(252)
                if compare_vals is not None:
                    compare_vals = (compare_vals[0] * 252, compare_vals[1] * np.sqrt(252), compare_vals[2], compare_vals[3])

            with sim_cols[0]:
                st.metric(label="Media simulada", value=f"{port_mean:.2f} %",
                          delta=f"{((port_mean - p_mean)/abs(p_mean)):.2f} %")
            with sim_cols[1]:
                st.metric(label="Desviación estándar simulada", value=f"{port_std:.2f} %",
                          delta=f"{((port_std - p_std)/abs(p_std)):.2f} %", delta_color="inverse")
            with sim_cols[2]:
                st.metric(label="Sharpe simulado", value=f"{port_sharpe:.2f}",
                          delta=f"{((port_sharpe - p_sharpe)/abs(p_sharpe)):.2f} %")
            with sim_cols[3]:
                st.metric(label="Alpha de Jensen simulado", value=f"{real_alpha:.4f}" if real_alpha is not None else "-",
                          delta=f"{((real_alpha - alpha)/abs(alpha)):.2f} %")


            st.subheader("Alpha y Betas en simulación vs estimados")
            prob_df = pd.DataFrame(prob_results)
            st.dataframe(prob_df.set_index("Parámetro").T, hide_index=False, use_container_width=True)

            # # ...existing code for displaying alpha/beta table...
            # st.subheader("Alpha y Betas en simulación")
            # sim_table = []
            # sim_table.append({
            #     "Tipo": "Estimado",
            #     "Alpha": f"{alpha:.4f}" if alpha is not None else "-",
            #     **{k: f"{betas[k]:.4f}" if betas and k in betas else "-" for k in compare_cols}
            # })
            # sim_table.append({
            #     "Tipo": "Real",
            #     "Alpha": f"{real_alpha:.4f}" if real_alpha is not None else "-",
            #     **{k: f"{real_betas[k]:.4f}" if real_betas and k in real_betas else "-" for k in compare_cols}
            # })
            # st.dataframe(pd.DataFrame(sim_table), hide_index=True, use_container_width=True)
        else:
            st.info("No hay datos de simulación disponibles para los tickers seleccionados.")
        

def tab_0(e_returns, prices=None, yearly=False, usd_amount=5000.0, sim_returns=None, compare_returns=None, compare_cols=['Mkt-RF', 'SMB', 'HML']):
    n = len(e_returns.columns)
    if n == 0:
        st.info("No hay tickers válidos realizar el análisis.")
        return

    weights = [1/n] * n
    p_mean, p_std, p_sharpe = get_mean_and_std(e_returns, weights)
    port_ret = (e_returns * weights).sum(axis=1)
    alpha, betas = get_alpha_beta(port_ret, compare_returns)
    show_portfolio_results(
        e_returns, weights, p_mean, p_std, p_sharpe,
        prices=prices, yearly=yearly, usd_amount=usd_amount,
        sim_returns=sim_returns, alpha=alpha, betas=betas, compare_cols=compare_cols, sim_plot_key="sim_shewhart_plot_0"
    )

    return p_mean, p_std, p_sharpe, alpha

def tab_1(e_returns, compare_vals, prices=None, yearly=False, usd_amount=5000.0, sim_returns=None, compare_returns=None, compare_cols=['Mkt-RF', 'SMB', 'HML']):
    n = len(e_returns.columns)
    if n == 0:
        st.info("No hay tickers válidos realizar el análisis.")
        return

    # Objective: minimize portfolio std
    def portfolio_std(weights):
        _, std, _ = get_mean_and_std(e_returns, weights)
        return std

    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    })
    bounds = [(0, 1)] * n
    initial_weights = np.array([1/n] * n)

    result = minimize(portfolio_std, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = result.x

    p_mean, p_std, p_sharpe = get_mean_and_std(e_returns, opt_weights)
    port_ret = (e_returns * opt_weights).sum(axis=1)
    alpha, betas = get_alpha_beta(port_ret, compare_returns)
    show_portfolio_results(
        e_returns, opt_weights, p_mean, p_std, p_sharpe,
        prices=prices,
        compare_vals=compare_vals,
        yearly=yearly,
        usd_amount=usd_amount,
        sim_returns=sim_returns,
        alpha=alpha,
        betas=betas,
        compare_cols=compare_cols,
        sim_plot_key="sim_shewhart_plot_1"
    )

def tab_2(e_returns, compare_vals, prices=None, yearly=False, usd_amount=5000.0, sim_returns=None, compare_returns=None, compare_cols=['Mkt-RF', 'SMB', 'HML']):
    n = len(e_returns.columns)
    if n == 0:
        st.info("No hay tickers válidos realizar el análisis.")
        return

    # Objective: maximize Sharpe ratio (minimize negative Sharpe)
    def neg_sharpe(weights):
        _, _, sharpe = get_mean_and_std(e_returns, weights)
        return -sharpe

    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    })
    bounds = [(0, 1)] * n
    initial_weights = np.array([1/n] * n)

    result = minimize(neg_sharpe, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = result.x

    p_mean, p_std, p_sharpe = get_mean_and_std(e_returns, opt_weights)
    port_ret = (e_returns * opt_weights).sum(axis=1)
    alpha, betas = get_alpha_beta(port_ret, compare_returns)
    show_portfolio_results(
        e_returns, opt_weights, p_mean, p_std, p_sharpe,
        prices=prices,
        compare_vals=compare_vals,
        yearly=yearly,
        usd_amount=usd_amount,
        sim_returns=sim_returns,
        alpha=alpha,
        betas=betas,
        compare_cols=compare_cols,
        sim_plot_key="sim_shewhart_plot_2"
    )

def tab_3(
    e_returns, compare_vals, prices=None, yearly=False, usd_amount=5000.0,
    sim_returns=None, compare_returns=None, compare_cols=['Mkt-RF', 'SMB', 'HML']
):
    n = len(e_returns.columns)
    if n == 0:
        st.info("No hay tickers válidos para realizar el análisis.")
        return

    # Objective: maximize Jensen's alpha
    def neg_alpha(weights):
        port_ret = (e_returns * weights).sum(axis=1)
        alpha, _ = get_alpha_beta(port_ret, compare_returns, x_cols=compare_cols)
        # If alpha is nan, penalize
        return -alpha if not np.isnan(alpha) else 1e6

    constraints = ({
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1
    })
    bounds = [(0, 1)] * n
    initial_weights = np.array([1/n] * n)

    result = minimize(neg_alpha, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_weights = result.x

    p_mean, p_std, p_sharpe = get_mean_and_std(e_returns, opt_weights)
    port_ret = (e_returns * opt_weights).sum(axis=1)
    alpha, betas = get_alpha_beta(port_ret, compare_returns, x_cols=compare_cols)
    show_portfolio_results(
        e_returns, opt_weights, p_mean, p_std, p_sharpe,
        prices=prices,
        compare_vals=compare_vals,
        yearly=yearly,
        usd_amount=usd_amount,
        sim_returns=sim_returns,
        alpha=alpha,
        betas=betas,
        compare_cols=compare_cols,
        sim_plot_key="sim_shewhart_plot_3"
    )

def get_weighted_portfolio_data(data, weights, tickers):
    # Calcula los retornos ponderados, media y desviación estándar
    data_weighted = data[tickers].copy()
    for ticker, weight in zip(tickers, weights):
        data_weighted[ticker] = data_weighted[ticker] * weight
    data_weighted["Portfolio"] = data_weighted.sum(axis=1)
    port_mean = data_weighted["Portfolio"].mean()
    port_std = data_weighted["Portfolio"].std(ddof=1)
    return data_weighted, port_mean, port_std

def shewheart_plot(data_weighted, port_mean, port_std, key=None):
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
        title="Gráfico de Shewhart para el periodo de simulación",
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

    st.plotly_chart(fig, use_container_width=True, key=key)

def get_alpha_beta(portfolio_returns, returns_df_analysis, x_cols=None):
    """
    Calculates Jensen's Alpha and market Beta(s) for the portfolio using OLS.
    portfolio_returns: pd.Series (index aligned with returns_df_analysis)
    returns_df_analysis: pd.DataFrame (columns to use as X)
    x_cols: list of column names to use as X (if None, use all columns in returns_df_analysis)
    Returns: (alpha, betas) where betas is a dict {col: value}
    """
    if x_cols is None:
        x_cols = list(returns_df_analysis.columns)
    # Ensure alignment and drop NaNs
    df = pd.DataFrame({"portfolio": portfolio_returns})
    for col in x_cols:
        df[col] = returns_df_analysis[col]
    df = df.dropna()
    if df.empty:
        return np.nan, {col: np.nan for col in x_cols}

    X = df[x_cols].values
    y = df["portfolio"].values

    # Add constant for intercept (alpha)
    X_mat = np.column_stack([np.ones(len(X)), X])
    # OLS solution: beta_hat = (X'X)^-1 X'y
    beta_hat = np.linalg.lstsq(X_mat, y, rcond=None)[0]
    alpha = beta_hat[0]
    betas = {col: beta_hat[i+1] for i, col in enumerate(x_cols)}
    return alpha, betas
