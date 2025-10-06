import streamlit as st
import yfinance as yf
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Advanced Option Pricing Calculator",
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
        'MARUTI.NS', 'ASIANPAINT.NS', 'SUNPHARMA.NS', 'ADANIENT.NS', 'TITAN.NS',
        'ULTRACEMCO.NS', 'WIPRO.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS',
        'M&M.NS', 'JSWSTEEL.NS', 'TATAMOTORS.NS', 'ADANIPORTS.NS', 'POWERGRID.NS',
        'BAJAJFINSV.NS', 'COALINDIA.NS', 'TATASTEEL.NS', 'INDUSINDBK.NS', 'HINDALCO.NS',
        'TECHM.NS', 'GRASIM.NS', 'CIPLA.NS', 'EICHERMOT.NS', 'DRREDDY.NS',
        'SBILIFE.NS', 'DIVISLAB.NS', 'HEROMOTOCO.NS', 'BRITANNIA.NS', 'APOLLOHOSP.NS',
        'SHRIRAMFIN.NS', 'HDFCLIFE.NS', 'BAJAJ-AUTO.NS', 'BPCL.NS', 'LTIM.NS'
    ]

# --- Core Calculation Functions ---

def _calculate_binomial_price(S, K, T, r, sigma, N, option_type):
    """
    Internal helper function to calculate ONLY the option price.
    This prevents the recursion error.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Price tree
    stock_tree = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        for i in range(j + 1):
            stock_tree[i, j] = S * (u**(j - i)) * (d**i)

    # Option payoff at maturity
    option_tree = np.zeros((N + 1, N + 1))
    if option_type == 'Call':
        option_tree[:, N] = np.maximum(0, stock_tree[:, N] - K)
    else: # 'Put'
        option_tree[:, N] = np.maximum(0, K - stock_tree[:, N])

    # Backward induction
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            option_tree[i, j] = np.exp(-r * dt) * (
                p * option_tree[i, j + 1] + (1 - p) * option_tree[i + 1, j + 1]
            )
            
    # Return the trees needed for Greeks calculation along with the price
    return option_tree, stock_tree

def calculate_option_data(S, K, T, r_percent, sigma_percent, N, option_type):
    """
    Main function to calculate the option price and ALL Greeks without recursion.
    """
    r = r_percent / 100
    sigma = sigma_percent / 100

    # --- 1. Calculate the main price and get the trees ---
    option_tree, stock_tree = _calculate_binomial_price(S, K, T, r, sigma, N, option_type)
    price = option_tree[0, 0]

    # --- 2. Calculate Greeks using the generated trees ---
    dt = T / N
    # Delta
    delta = (option_tree[0, 1] - option_tree[1, 1]) / (stock_tree[0, 1] - stock_tree[1, 1])
    # Gamma
    delta_up = (option_tree[0, 2] - option_tree[1, 2]) / (stock_tree[0, 2] - stock_tree[1, 2])
    delta_down = (option_tree[1, 2] - option_tree[2, 2]) / (stock_tree[1, 2] - stock_tree[2, 2])
    gamma = (delta_up - delta_down) / (stock_tree[0, 2] - stock_tree[2, 2])
    # Theta (per day)
    theta = (option_tree[1, 2] - price) / (2 * dt) / 365

    # --- 3. Calculate Vega and Rho by calling the helper function ---
    # Vega (price change for a 1% change in volatility)
    vega_option_tree, _ = _calculate_binomial_price(S, K, T, r, sigma + 0.01, N, option_type)
    vega = vega_option_tree[0, 0] - price
    
    # Rho (price change for a 1% change in risk-free rate)
    rho_option_tree, _ = _calculate_binomial_price(S, K, T, r + 0.01, sigma, N, option_type)
    rho = rho_option_tree[0, 0] - price

    return {
        'price': price, 'delta': delta, 'gamma': gamma,
        'theta': theta, 'vega': vega, 'rho': rho
    }


# --- Streamlit UI (No changes needed below this line) ---
st.title("📈 Advanced Binomial Option Pricing Calculator")
st.markdown("For Nifty 50 Stocks | Calculates European Option Prices & Greeks")

st.sidebar.header("Input Parameters")
ticker = st.sidebar.selectbox("Select Nifty 50 Stock", get_nifty_50_tickers(), index=0)

try:
    stock_data = yf.Ticker(ticker)
    info = stock_data.info
    live_price = info.get('regularMarketPrice', info.get('previousClose'))
    if live_price is None:
        st.sidebar.error("Could not fetch price. Try another ticker.")
    else:
        st.sidebar.metric(label=f"Current Price for {ticker}", value=f"₹{live_price:,.2f}")
except Exception as e:
    st.sidebar.error(f"Failed to fetch data: {e}")
    live_price = None

option_type = st.sidebar.radio("Option Type", ('Call', 'Put'))
default_strike = float(round(live_price / 50) * 50) if live_price else 2000.0
strike_price = st.sidebar.number_input("Strike Price (K)", min_value=0.0, value=default_strike, step=10.0)
risk_free_rate = st.sidebar.number_input("Risk-Free Rate (Rf) in %", min_value=0.0, value=7.0, step=0.1)
volatility = st.sidebar.number_input("Implied Volatility (σ) in %", min_value=0.1, value=20.0, step=0.5)
time_to_exp = st.sidebar.number_input("Time to Expiration (Years, T)", min_value=0.01, value=0.25, step=0.01)
steps = st.sidebar.slider("Number of Steps (N)", min_value=10, max_value=200, value=100, step=10, help="Higher steps increase accuracy but take longer to compute.")

if st.sidebar.button("Calculate", use_container_width=True, type="primary"):
    if live_price is not None:
        with st.spinner('Calculating... Please wait.'):
            # Call the new, safe function
            results = calculate_option_data(
                live_price, strike_price, time_to_exp, risk_free_rate, volatility, steps, option_type
            )
        
        st.subheader(f"Results for {option_type} Option")
        st.metric(
            label=f"Theoretical {option_type} Option Price",
            value=f"₹{results['price']:,.2f}",
        )
        st.markdown("---")
        st.subheader("Option Greeks (Sensitivities)")
        col1, col2, col3 = st.columns(3)
        col1.metric(
            label="Delta (Δ)",
            value=f"{results['delta']:.4f}",
            help="Rate of change of the option price with respect to a ₹1 change in the underlying stock price."
        )
        col2.metric(
            label="Gamma (Γ)",
            value=f"{results['gamma']:.4f}",
            help="Rate of change in Delta with respect to a ₹1 change in the underlying stock price."
        )
        col3.metric(
            label="Theta (Θ)",
            value=f"₹{results['theta']:.4f} / day",
            help="Rate of change of the option price with respect to the passage of time (time decay)."
        )
        col4, col5, _ = st.columns(3)
        col4.metric(
            label="Vega (ν)",
            value=f"₹{results['vega']:.4f}",
            help="Rate of change of the option price with respect to a 1% change in volatility."
        )
        col5.metric(
            label="Rho (ρ)",
            value=f"₹{results['rho']:.4f}",
            help="Rate of change of the option price with respect to a 1% change in the risk-free rate."
        )
    else:
        st.error("Cannot calculate without a valid live stock price.")
else:
    st.info("Select parameters in the sidebar and click 'Calculate' to see the results.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Disclaimer: Educational tool for demonstrating the Binomial model. Not financial advice.</p>", unsafe_allow_html=True)
