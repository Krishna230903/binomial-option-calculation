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
    """Returns a list of Nifty 50 stock tickers."""
    return [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS',
        'BHARTIARTL.NS', 'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'LICI.NS',
        'BAJFINANCE.NS', 'HCLTECH.NS', 'KOTAKBANK.NS', 'LT.NS', 'AXISBANK.NS',
        'MARUTI.NS', 'ASIANPAINT.NS', 'SUNPHARMA.NS', 'ADANIENT.NS', 'TITAN.NS'
    ]

# --- Core Calculation Functions ---
def _calculate_binomial_price(S, K, T, r, sigma, N, option_type):
    """Helper function to build the binomial tree and calculate the price."""
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Arbitrage and probability check
    if u <= d or p < 0 or p > 1:
        error_msg = "Input parameters lead to arbitrage or invalid probabilities. Please adjust volatility or risk-free rate."
        return None, None, error_msg

    # Initialize asset price tree
    stock_tree = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        for i in range(j + 1):
            stock_tree[i, j] = S * (u**(j - i)) * (d**i)

    # Initialize option price tree and calculate terminal values
    option_tree = np.zeros((N + 1, N + 1))
    if option_type == 'Call':
        option_tree[:, N] = np.maximum(0, stock_tree[:, N] - K)
    else: # Put
        option_tree[:, N] = np.maximum(0, K - stock_tree[:, N])

    # Backward induction for option price
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            option_tree[i, j] = np.exp(-r * dt) * (
                p * option_tree[i, j + 1] + (1 - p) * option_tree[i + 1, j + 1]
            )

    return option_tree, stock_tree, None # Return None for error on success

@st.cache_data
def calculate_option_data(S, K, T, r_percent, sigma_percent, N, option_type):
    """
    Main function to calculate option price and all Greeks.
    This function is cached for performance.
    """
    r = r_percent / 100
    sigma = sigma_percent / 100

    option_tree, stock_tree, error = _calculate_binomial_price(S, K, T, r, sigma, N, option_type)

    if error:
        return {'error': error}

    price = option_tree[0, 0]
    dt = T / N

    # Greeks Calculation (requires at least 2 steps)
    if N < 3:
        delta, gamma, theta = 0, 0, 0
    else:
        # Delta: Change in option price for a change in stock price
        delta = (option_tree[0, 1] - option_tree[1, 1]) / (stock_tree[0, 1] - stock_tree[1, 1])
        # Gamma: Change in delta for a change in stock price
        delta_up = (option_tree[0, 2] - option_tree[1, 2]) / (stock_tree[0, 2] - stock_tree[1, 2])
        delta_down = (option_tree[1, 2] - option_tree[2, 2]) / (stock_tree[1, 2] - stock_tree[2, 2])
        gamma = (delta_up - delta_down) / (stock_tree[0, 2] - stock_tree[2, 2])
        # Theta: Time decay, per day
        theta = (option_tree[1, 2] - price) / (2 * dt) / 365

    # Vega & Rho (calculated by bumping inputs)
    vega_option_tree, _, vega_error = _calculate_binomial_price(S, K, T, r, sigma + 0.01, N, option_type)
    rho_option_tree, _, rho_error = _calculate_binomial_price(S, K, T, r + 0.01, sigma, N, option_type)

    vega = vega_option_tree[0, 0] - price if not vega_error else 0
    rho = rho_option_tree[0, 0] - price if not rho_error else 0

    return {
        'price': price, 'delta': delta, 'gamma': gamma,
        'theta': theta, 'vega': vega, 'rho': rho
    }

# --- Streamlit UI ---
st.title("📈 Live Option Pricing & Scenario Analysis")
st.markdown("An interactive tool using the **Cox-Ross-Rubinstein Binomial Model**.")

# --- Sidebar for Inputs ---
st.sidebar.title("🔢 Input Parameters")
ticker = st.sidebar.selectbox("Select Nifty 50 Stock", get_nifty_50_tickers(), index=0)

try:
    stock_data = yf.Ticker(ticker)
    info = stock_data.info
    live_price = info.get('regularMarketPrice', info.get('previousClose'))
    if live_price is None:
        st.sidebar.error("Could not fetch a valid price for this ticker.")
    else:
        st.sidebar.metric(label=f"Current Price for {ticker}", value=f"₹{live_price:,.2f}")
except Exception as e:
    st.sidebar.error(f"Failed to fetch data: {e}")
    live_price = None

option_type = st.sidebar.radio("Option Type", ('Call', 'Put'), horizontal=True)

col1, col2 = st.sidebar.columns(2)
default_strike = float(round(live_price / 50) * 50) if live_price else 3000.0
strike_price = col1.number_input("Strike Price (K)", min_value=0.01, value=default_strike, step=10.0)
time_to_exp_days = col2.number_input("Expiry (Days)", min_value=1, value=30, step=1)
time_to_exp = time_to_exp_days / 365.0

risk_free_rate = st.sidebar.slider("Risk-Free Rate (Rf) %", min_value=0.0, max_value=15.0, value=7.0, step=0.1)
volatility = st.sidebar.slider("Implied Volatility (σ) %", min_value=1.0, max_value=100.0, value=20.0, step=0.5)
steps = st.sidebar.slider("Model Steps (N)", min_value=10, max_value=500, value=100, step=10, help="Higher steps increase accuracy but take longer to compute.")

# --- Calculation and Display Logic ---
if st.sidebar.button("Calculate", use_container_width=True, type="primary"):
    if live_price is not None:
        with st.spinner('Running calculations and generating analysis...'):
            results = calculate_option_data(live_price, strike_price, time_to_exp, risk_free_rate, volatility, steps, option_type)

            # Check for errors from the calculation
            if 'error' in results:
                st.error(f"Calculation Error: {results['error']}")
            else:
                # --- Main Display Area ---
                st.subheader(f"Valuation for {ticker} {option_type} Option")
                res_col1, res_col2 = st.columns([1, 2])
                res_col1.metric(label="Theoretical Option Price", value=f"₹{results.get('price', 0):,.2f}")
                res_col2.markdown(f"""
                - **Underlying Price (S):** `₹{live_price:,.2f}`
                - **Strike Price (K):** `₹{strike_price:,.2f}`
                - **Days to Expiry (T):** `{time_to_exp_days}`
                """)
                st.markdown("---")

                # --- Greeks Analysis in Tabs ---
                st.subheader("🔬 Option Greeks Analysis")
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Delta (Δ)", "Gamma (Γ)", "Theta (Θ)", "Vega (ν)", "Rho (ρ)"])
                with tab1:
                    st.metric(label="Delta Value", value=f"{results.get('delta', 0):.4f}", help="Measures the rate of change of the option price with respect to a ₹1 change in the underlying asset's price.")
                with tab2:
                    st.metric(label="Gamma Value", value=f"{results.get('gamma', 0):.4f}", help="Measures the rate of change in Delta for a ₹1 change in the underlying price. Represents the convexity of the option's value.")
                with tab3:
                    st.metric(label="Theta Value (per day)", value=f"₹{results.get('theta', 0):.4f}", help="Measures the rate of decline in the value of an option due to the passage of time (time decay).")
                with tab4:
                    st.metric(label="Vega Value", value=f"₹{results.get('vega', 0):,.2f}", help="Measures the change in an option's price for a 1% change in the implied volatility of the underlying stock.")
                with tab5:
                    st.metric(label="Rho Value", value=f"₹{results.get('rho', 0):,.2f}", help="Measures the change in an option's price for a 1% change in the risk-free interest rate.")

                # --- Live Quantitative Analysis Report ---
                with st.expander("📝 Live Quantitative Analysis Report", expanded=True):
                    st.markdown("This report shows how the option's value changes when one variable is adjusted while others are held constant.")

                    # 1. Underlying Price Analysis
                    st.markdown("#### 1. Impact of Underlying Asset Price (S)")
                    s_range = np.linspace(live_price * 0.9, live_price * 1.1, 10)
                    price_analysis_data = [calculate_option_data(s, strike_price, time_to_exp, risk_free_rate, volatility, steps, option_type).get('price', 0) for s in s_range]
                    price_df = pd.DataFrame({'Underlying Price': s_range, f'{option_type} Price': price_analysis_data}).set_index('Underlying Price')
                    st.line_chart(price_df)
                    st.markdown(f"**Analysis:** The chart shows the direct relationship between the stock price and the {option_type.lower()} option's value. The slope of this curve at any point is the **Delta**.")

                    # 2. Volatility Analysis
                    st.markdown("#### 2. Impact of Volatility (σ)")
                    vol_range = np.linspace(max(1.0, volatility - 10), volatility + 10, 10)
                    vol_analysis_data = [calculate_option_data(live_price, strike_price, time_to_exp, risk_free_rate, vol, steps, option_type).get('price', 0) for vol in vol_range]
                    vol_df = pd.DataFrame({'Implied Volatility (%)': vol_range, f'{option_type} Price': vol_analysis_data}).set_index('Implied Volatility (%)')
                    st.line_chart(vol_df)
                    st.markdown(f"**Analysis:** Higher volatility increases the price of both calls and puts because it raises the probability of the option finishing deep in-the-money. **Vega** measures this sensitivity.")

                    # 3. Time to Expiration Analysis
                    st.markdown("#### 3. Impact of Time to Expiration (T)")
                    time_range_days = np.linspace(max(1, time_to_exp_days - 25), time_to_exp_days + 30, 10, dtype=int)
                    time_analysis_data = [calculate_option_data(live_price, strike_price, t/365.0, risk_free_rate, volatility, steps, option_type).get('price', 0) for t in time_range_days]
                    time_df = pd.DataFrame({'Days to Expiry': time_range_days, f'{option_type} Price': time_analysis_data}).set_index('Days to Expiry')
                    st.line_chart(time_df)
                    st.markdown(f"**Analysis:** The option loses value as time passes, and the rate of this decay (Theta) accelerates as expiration approaches. This phenomenon is known as **time decay**.")

    else:
        st.error("Cannot perform calculation without a valid live stock price. Please check the ticker selection.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Disclaimer: This is an educational tool for demonstrating financial models. Not financial advice.</p>", unsafe_allow_html=True)
