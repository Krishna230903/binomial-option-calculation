import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="Live Option Pricing & Analysis",
    page_icon="📈",
    layout="wide"
)

# --- NIFTY 50 Tickers ---
@st.cache_data
def get_nifty_50_tickers():
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
        'BHARTIARTL.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'LICI.NS',
        'BAJFINANCE.NS', 'HCLTECH.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS',
        'MARUTI.NS', 'ASIANPAINT.NS', 'SUNPHARMA.NS', 'ADANIENT.NS', 'TITAN.NS'
    ]

# --- Core Calculation Functions ---
def _calculate_binomial_price(S, K, T, r, sigma, N, option_type):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    stock_tree = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        for i in range(j + 1):
            stock_tree[i, j] = S * (u**(j - i)) * (d**i)
    option_tree = np.zeros((N + 1, N + 1))
    if option_type == 'Call':
        option_tree[:, N] = np.maximum(0, stock_tree[:, N] - K)
    else:
        option_tree[:, N] = np.maximum(0, K - stock_tree[:, N])
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            option_tree[i, j] = np.exp(-r * dt) * (
                p * option_tree[i, j + 1] + (1 - p) * option_tree[i + 1, j + 1]
            )
    return option_tree, stock_tree

def calculate_option_data(S, K, T, r_percent, sigma_percent, N, option_type):
    r = r_percent / 100
    sigma = sigma_percent / 100
    # Add a check for valid inputs to prevent math errors
    if u <= d or p < 0 or p > 1:
        return {'error': "Input parameters lead to arbitrage or invalid probabilities. Please adjust."}
        
    option_tree, stock_tree = _calculate_binomial_price(S, K, T, r, sigma, N, option_type)
    price = option_tree[0, 0]
    dt = T / N

    # Greeks calculation with safety checks for small N
    if N < 3:
        delta, gamma, theta = 0, 0, 0
    else:
        delta = (option_tree[0, 1] - option_tree[1, 1]) / (stock_tree[0, 1] - stock_tree[1, 1])
        delta_up = (option_tree[0, 2] - option_tree[1, 2]) / (stock_tree[0, 2] - stock_tree[1, 2])
        delta_down = (option_tree[1, 2] - option_tree[2, 2]) / (stock_tree[1, 2] - stock_tree[2, 2])
        gamma = (delta_up - delta_down) / (stock_tree[0, 2] - stock_tree[2, 2])
        theta = (option_tree[1, 2] - price) / (2 * dt) / 365
        
    # Vega & Rho
    vega_option_tree, _ = _calculate_binomial_price(S, K, T, r, sigma + 0.01, N, option_type)
    vega = vega_option_tree[0, 0] - price
    rho_option_tree, _ = _calculate_binomial_price(S, K, T, r + 0.01, sigma, N, option_type)
    rho = rho_option_tree[0, 0] - price

    return {
        'price': price, 'delta': delta, 'gamma': gamma,
        'theta': theta, 'vega': vega, 'rho': rho
    }

# --- Streamlit UI ---
st.title("📈 Live Option Pricing & Scenario Analysis")
st.markdown("An interactive tool using the **Cox-Ross-Rubinstein Binomial Model**.")

st.sidebar.title("🔢 Input Parameters")
ticker = st.sidebar.selectbox("Select Nifty 50 Stock", get_nifty_50_tickers(), index=0)

try:
    stock_data = yf.Ticker(ticker)
    info = stock_data.info
    live_price = info.get('regularMarketPrice', info.get('previousClose'))
    if live_price is None: st.sidebar.error("Could not fetch price.")
    else: st.sidebar.metric(label=f"Current Price for {ticker}", value=f"₹{live_price:,.2f}")
except Exception as e:
    st.sidebar.error(f"Failed to fetch data: {e}")
    live_price = None

option_type = st.sidebar.radio("Option Type", ('Call', 'Put'), horizontal=True)
col1, col2 = st.sidebar.columns(2)
default_strike = float(round(live_price / 50) * 50) if live_price else 3000.0
strike_price = col1.number_input("Strike Price (K)", min_value=0.0, value=default_strike, step=10.0)
time_to_exp_days = col2.number_input("Expiry (Days)", min_value=1, value=30, step=1)
time_to_exp = time_to_exp_days / 365.0

risk_free_rate = st.sidebar.number_input("Risk-Free Rate (Rf) in %", min_value=0.0, value=7.0, step=0.1)
volatility = st.sidebar.number_input("Implied Volatility (σ) in %", min_value=0.1, value=20.0, step=0.5)
steps = st.sidebar.slider("Model Steps (N)", min_value=10, max_value=500, value=100, step=10, help="Higher steps increase accuracy.")

if st.sidebar.button("Calculate", use_container_width=True, type="primary"):
    if live_price is not None:
        with st.spinner('Running calculations and live analysis...'):
            results = calculate_option_data(live_price, strike_price, time_to_exp, risk_free_rate, volatility, steps, option_type)
            
            # --- Main Display Area ---
            st.subheader(f"Valuation for {ticker} {option_type} Option")
            res_col1, res_col2 = st.columns([1, 2])
            res_col1.metric(label=f"Theoretical Option Price", value=f"₹{results.get('price', 0):,.2f}")
            res_col2.markdown(f"""
            - **Underlying Price (S):** `₹{live_price:,.2f}`
            - **Strike Price (K):** `₹{strike_price:,.2f}`
            - **Days to Expiry (T):** `{time_to_exp_days}`
            """)
            st.markdown("---")
            
            # --- Greeks Analysis in Tabs ---
            st.subheader("🔬 Option Greeks Analysis")
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Delta (Δ)", "Gamma (Γ)", "Theta (Θ)", "Vega (ν)", "Rho (ρ)"])
            # (Tabs content as before, referencing the 'results' dictionary)
            with tab1: st.metric(label="Delta Value", value=f"{results.get('delta', 0):.4f}")
            with tab2: st.metric(label="Gamma Value", value=f"{results.get('gamma', 0):.4f}")
            with tab3: st.metric(label="Theta Value (per day)", value=f"₹{results.get('theta', 0):.4f}")
            with tab4: st.metric(label="Vega Value", value=f"₹{results.get('vega', 0):.4f}")
            with tab5: st.metric(label="Rho Value", value=f"₹{results.get('rho', 0):.4f}")

            # --- Live Quantitative Analysis Report ---
            with st.expander("📝 Live Quantitative Analysis Report", expanded=True):
                st.markdown("This report is dynamically generated based on your inputs. It shows how the option's value changes when one variable is adjusted while others are held constant.")
                
                # 1. Underlying Price Analysis
                st.markdown("#### 1. Impact of Underlying Asset Price (S)")
                price_data = []
                s_range = [live_price * 0.95, live_price, live_price * 1.05]
                for s_val in s_range:
                    res = calculate_option_data(s_val, strike_price, time_to_exp, risk_free_rate, volatility, steps, option_type)
                    price_data.append({'Underlying Price (S)': f"₹{s_val:,.2f}", f'{option_type} Price': f"₹{res.get('price', 0):,.2f}", 'Delta (Δ)': f"{res.get('delta', 0):.3f}"})
                price_df = pd.DataFrame(price_data)
                st.dataframe(price_df, use_container_width=True)
                st.markdown(f"**Analysis:** As the stock price moves from `₹{s_range[0]:,.2f}` to `₹{s_range[2]:,.2f}`, the {option_type.lower()} price changes significantly. The **Delta** quantifies this sensitivity, showing how the option's responsiveness changes with its 'moneyness'.")

                # 2. Volatility Analysis
                st.markdown("#### 2. Impact of Volatility (σ)")
                vol_data = []
                vol_range = [max(1.0, volatility - 5), volatility, volatility + 5]
                for vol_val in vol_range:
                    res = calculate_option_data(live_price, strike_price, time_to_exp, risk_free_rate, vol_val, steps, option_type)
                    vol_data.append({'Volatility (σ)': f"{vol_val:.1f}%", f'{option_type} Price': f"₹{res.get('price', 0):,.2f}", 'Vega (ν)': f"₹{res.get('vega', 0):,.2f}"})
                vol_df = pd.DataFrame(vol_data)
                st.dataframe(vol_df, use_container_width=True)
                st.markdown(f"**Analysis:** Changing volatility from `{vol_range[0]:.1f}%` to `{vol_range[2]:.1f}%` has a strong impact on the option price. The **Vega** of `₹{results.get('vega', 0):,.2f}` (from your base calculation) indicates the approximate price increase for each 1% rise in volatility.")
                
                # 3. Time to Expiration Analysis
                st.markdown("#### 3. Impact of Time to Expiration (T)")
                time_data = []
                time_range_days = [time_to_exp_days + 30, time_to_exp_days, max(1, time_to_exp_days - 15)]
                for t_days in time_range_days:
                    t_years = t_days / 365.0
                    res = calculate_option_data(live_price, strike_price, t_years, risk_free_rate, volatility, steps, option_type)
                    time_data.append({'Days to Expiry': t_days, f'{option_type} Price': f"₹{res.get('price', 0):,.2f}", 'Theta (Θ) per day': f"₹{res.get('theta', 0):,.2f}"})
                time_df = pd.DataFrame(time_data)
                st.dataframe(time_df, use_container_width=True)
                st.markdown(f"**Analysis:** This table clearly illustrates time decay. An option with `{time_range_days[0]}` days of life is worth more than one with `{time_range_days[2]}` days. The **Theta** shows that the rate of daily value loss accelerates as the option gets closer to expiration.")
                
    else:
        st.error("Cannot calculate without a valid live stock price.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Disclaimer: Educational tool. Not financial advice.</p>", unsafe_allow_html=True)
