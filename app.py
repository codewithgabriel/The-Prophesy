import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
from envs.config import CONFIG
from brokers.broker_alpaca import AlpacaBroker
from brokers.broker_ccxt import CCXTBroker
from models.train_ppo import train_ppo_model
from utils import run_backtest, create_env, load_and_prepare_data, load_model, plot_equity_curve, plot_trades

# --- Page Setup ---
st.set_page_config(
    layout="wide",
    page_title="RL Trading Dashboard",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1E88E5; text-align: center; margin-bottom: 1rem; font-weight: 700; }
    .metric-card { background-color: #f9f9f9; padding: 1rem; border-radius: 0.5rem; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .positive-value { color: #00C853; font-weight: 700; }
    .negative-value { color: #FF5252; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# --- Session State Init ---
for key, default in {
    'training_progress': 0,
    'training_status': "Not started",
    'backtest_results': None,
    'trade_decisions': [],
    'current_balance': CONFIG["initial_balance"],
    'balance_history': []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --- Sidebar Navigation ---
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio("Go to:", ["Dashboard", "Backtest", "Live Trading", "Model Training"],
                        format_func=lambda x: {
                            "Dashboard": "📊 Dashboard",
                            "Backtest": "🔁 Backtest",
                            "Live Trading": "🔴 Live Trading",
                            "Model Training": "🤖 Model Training"
                        }[x])

# ================= Dashboard =================
if menu == "Dashboard":
    st.markdown('<h1 class="main-header">📈 Professional RL Trading Dashboard</h1>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Balance", f"${st.session_state.current_balance:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        pnl = st.session_state.current_balance - CONFIG["initial_balance"]
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Profit/Loss", f"${pnl:,.2f}", delta=f"{pnl/CONFIG['initial_balance']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Trades", str(len(st.session_state.trade_decisions)))
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        win_rate = len([t for t in st.session_state.trade_decisions if t.get('profit',0) > 0])
        win_rate = win_rate / len(st.session_state.trade_decisions) * 100 if st.session_state.trade_decisions else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Win Rate", f"{win_rate:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.backtest_results:
        networth, trades = st.session_state.backtest_results
        st.plotly_chart(plot_equity_curve(networth, CONFIG["initial_balance"]), use_container_width=True)

    if st.session_state.trade_decisions:
        st.subheader("Recent Trading Decisions")
        df = pd.DataFrame(st.session_state.trade_decisions[-10:])
        if 'timestamp' in df:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
        if 'position_shares' in df:
            df['action'] = df['position_shares'].apply(lambda x: 'BUY' if x>0 else 'SELL' if x<0 else 'HOLD')
        st.dataframe(df, use_container_width=True)

# ================= Backtest =================
elif menu == "Backtest":
    st.header("🔁 Backtesting")

    start_date = st.date_input("Start Date", datetime(2020,1,1))
    end_date = st.date_input("End Date", datetime.today())

    if st.button("Run Backtest"):
        try:
            train_df, test_df = load_and_prepare_data(start_date, end_date)
            model = load_model()
            networth, trades = run_backtest(model, test_df)
            st.session_state.backtest_results = (networth, trades)
            st.session_state.trade_decisions = trades
            st.session_state.balance_history = networth.tolist()
            st.session_state.current_balance = networth[-1]
            st.success("Backtest completed successfully!")
        except Exception as e:
            st.error(f"Backtest failed: {e}")

    if st.session_state.backtest_results:
        networth, trades = st.session_state.backtest_results
        tab1, tab2 = st.tabs(["Equity Curve", "Trades"])
        with tab1:
            st.plotly_chart(plot_equity_curve(networth, CONFIG["initial_balance"]), use_container_width=True)
        with tab2:
            train_df, test_df = load_and_prepare_data(start_date, end_date)
            st.plotly_chart(plot_trades(test_df, trades), use_container_width=True)

# ================= Live Trading =================
elif menu == "Live Trading":
    st.header("🔴 Live Trading")

    broker_type = st.selectbox("Broker", ["Alpaca", "Binance (CCXT)"])
    api_key = st.text_input("API Key")
    api_secret = st.text_input("API Secret", type="password")
    start_date = st.date_input("Start Date", datetime(2022,1,1))
    end_date = st.date_input("End Date", datetime.today())

    if broker_type == "Alpaca":
        base_url = st.text_input("Base URL", "https://paper-api.alpaca.markets")
        if st.button("Connect to Alpaca"):
            try:
                broker = AlpacaBroker(api_key, api_secret, base_url)
                st.success("Connected!")
                st.json(broker.get_account())
            except Exception as e:
                st.error(f"Connection failed: {e}")
    else:
        if st.button("Connect to Binance"):
            try:
                broker = CCXTBroker("binance", api_key, api_secret)
                st.success("Connected!")
                st.json(broker.get_balance())
            except Exception as e:
                st.error(f"Connection failed: {e}")

# ================= Model Training =================
elif menu == "Model Training":
    st.header("🤖 Model Training")
    st.info("Set training params below")

    lr = st.slider("Learning Rate", 0.0001, 0.01, 0.0003, 0.0001)
    steps = st.slider("Training Steps", 1000, 100000, 10000, 1000)
    if st.button("Start Training"):
        try:
            train_df, test_df = load_and_prepare_data()
            env, eval_env = create_env(train_df, test_df)
            model = train_ppo_model(env, eval_env)
            st.success("Model trained successfully!")
        except Exception as e:
            st.error(f"Training failed: {e}")
