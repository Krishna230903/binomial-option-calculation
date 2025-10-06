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
    option_tree, stock_tree = _calculate_binomial_price(S, K, T, r, sigma, N, option_type)
    price = option_tree[0, 0]
    dt = T / N
    delta = (option_tree[0, 1] - option_tree[1, 1]) / (stock_tree[0, 1] - stock_tree[1, 1])
    delta_up = (option_tree[0, 2] - option_tree[1, 2]) / (stock_tree[0, 2] - stock_tree[1, 2])
    delta_down = (option_tree[1, 2] - option_tree[2, 2]) / (stock_tree[1, 2] - stock_tree[2, 2])
    gamma = (delta_up - delta_down) / (stock_tree[0, 2] - stock_tree[2, 2])
    theta = (option_tree[1, 2] - price) / (2 * dt) / 365
    vega_option_tree, _ = _calculate_binomial_price(S, K, T, r, sigma + 0.01, N, option_type)
    vega = vega_option_tree[0, 0] - price
    rho_option_tree, _ = _calculate_binomial_price(S, K, T, r + 0.01, sigma, N, option_type)
    rho = rho_option_tree[0, 0] - price
    return {
        'price': price, 'delta': delta, 'gamma': gamma,
        'theta': theta, 'vega': vega, 'rho': rho
    }

# --- Streamlit UI ---
st.title("📈 Advanced Binomial Option Pricing Calculator")
st.markdown("For Nifty 50 Stocks | Calculates European Option Prices & Greeks using the Cox-Ross-Rubinstein Model.")

st.sidebar.title("🔢 Input Parameters")
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

option_type = st.sidebar.radio("Option Type", ('Call', 'Put'), horizontal=True)
col1, col2 = st.sidebar.columns(2)
default_strike = float(round(live_price / 50) * 50) if live_price else 2000.0
strike_price = col1.number_input("Strike Price (K)", min_value=0.0, value=default_strike, step=10.0)
time_to_exp_days = col2.number_input("Expiry (Days)", min_value=1, value=30, step=1)
time_to_exp = time_to_exp_days / 365.0

risk_free_rate = st.sidebar.number_input("Risk-Free Rate (Rf) in %", min_value=0.0, value=7.0, step=0.1)
volatility = st.sidebar.number_input("Implied Volatility (σ) in %", min_value=0.1, value=20.0, step=0.5)
steps = st.sidebar.slider("Model Steps (N)", min_value=10, max_value=500, value=100, step=10, help="Higher steps increase accuracy but take longer to compute.")

if st.sidebar.button("Calculate", use_container_width=True, type="primary"):
    if live_price is not None:
        with st.spinner('Building the binomial tree and calculating...'):
            results = calculate_option_data(
                live_price, strike_price, time_to_exp, risk_free_rate, volatility, steps, option_type
            )
        
        st.subheader(f"Valuation for {ticker} {option_type} Option")
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(label=f"Theoretical Option Price", value=f"₹{results['price']:,.2f}")
        with res_col2:
            st.markdown(f"""
            - **Underlying Price (S):** `₹{live_price:,.2f}`
            - **Strike Price (K):** `₹{strike_price:,.2f}`
            - **Days to Expiry (T):** `{time_to_exp_days}`
            """)
        
        st.markdown("---")
        
        # --- NEW: Greeks Displayed in Tabs ---
        st.subheader("🔬 Option Greeks Analysis")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Delta (Δ)", "Gamma (Γ)", "Theta (Θ)", "Vega (ν)", "Rho (ρ)"])

        with tab1:
            st.metric(label="Delta Value", value=f"{results['delta']:.4f}")
            st.markdown("#### How Delta is Calculated")
            st.markdown("Delta measures the change in the option price for a ₹1 change in the underlying stock. It's calculated from the first step of the binomial tree.")
            st.latex(r'''
            \Delta = \frac{C_{u} - C_{d}}{S_{u} - S_{d}}
            ''')
            st.markdown(r"""
            - $C_{u}$ and $C_{d}$ are the option prices at the first 'up' and 'down' nodes.
            - $S_{u}$ and $S_{d}$ are the stock prices at the first 'up' and 'down' nodes.
            """)

        with tab2:
            st.metric(label="Gamma Value", value=f"{results['gamma']:.4f}")
            st.markdown("#### How Gamma is Calculated")
            st.markdown("Gamma measures the rate of change of Delta. It's highest for at-the-money options and is calculated using values from the first two steps of the tree.")
            st.latex(r'''
            \Gamma = \frac{\Delta_{u} - \Delta_{d}}{S_{uu} - S_{dd}}
            ''')
            st.markdown(r"""
            - $\Delta_{u}$ is the Delta of the 'up' branch.
            - $\Delta_{d}$ is the Delta of the 'down' branch.
            - $S_{uu}$ and $S_{dd}$ are stock prices after two 'up' or two 'down' moves.
            """)

        with tab3:
            st.metric(label="Theta Value (per day)", value=f"₹{results['theta']:.4f}")
            st.markdown("#### How Theta is Calculated")
            st.markdown("Theta represents the option's time decay. It's calculated by measuring the change in the option's price over the first two time steps of the model.")
            st.latex(r'''
            \Theta = \frac{C_{1,2} - C_{0,0}}{2 \Delta t}
            ''')
            st.markdown(r"""
            - $C_{1,2}$ is the option price at the middle node two steps into the future.
            - $C_{0,0}$ is the option price today.
            - $\Delta t$ is the duration of a single time step.
            """)

        with tab4:
            st.metric(label="Vega Value", value=f"₹{results['vega']:.4f}")
            st.markdown("#### How Vega is Calculated")
            st.markdown("Vega is found numerically. The model recalculates the entire option price with a 1% increase in volatility (`σ`) and measures the difference.")
            st.latex(r'''
            \nu = \text{Price}(\sigma + 1\%) - \text{Price}(\sigma)
            ''')
            
        with tab5:
            st.metric(label="Rho Value", value=f"₹{results['rho']:.4f}")
            st.markdown("#### How Rho is Calculated")
            st.markdown("Similar to Vega, Rho is found numerically. The model recalculates the option price with a 1% increase in the risk-free rate (`r`) and measures the difference.")
            st.latex(r'''
            \rho = \text{Price}(r + 1\%) - \text{Price}(r)
            ''')

    else:
        st.error("Cannot calculate without a valid live stock price.")

with st.expander("📘 View Overall Model Explanations"):
    # (The general explanation from the previous version can remain here)
    st.markdown("...") # Omitted for brevity, but you can paste the previous explanation here

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>Disclaimer: Educational tool. Not financial advice.</p>", unsafe_allow_html=True)
