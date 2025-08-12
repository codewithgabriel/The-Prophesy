# ========================================= =======
# 📂 trading_app/main.py
# ================================================
import streamlit as st
from envs.config import CONFIG
from utils import load_data, save_trades_to_csv
from envs.trading_env import TradingEnv
from  models.model import train_model, load_model
from dashboard import plot_equity_curve, plot_trades
from brokers.broker_alpaca import AlpacaBroker
from brokers.broker_ccxt import CCXTBroker

st.set_page_config(layout="wide")
st.title("📈 RL Trading Dashboard")

menu = st.sidebar.selectbox("Menu", ["Backtest", "Live Trading"])

if menu == "Backtest":
    df = load_data(CONFIG["csv_path"], CONFIG["start_date"], CONFIG["end_date"])
    env = TradingEnv(df, CONFIG)
    if st.sidebar.button("Train Model"):
        model = train_model(env, CONFIG)
        st.success("Model trained and saved!")
    if st.sidebar.button("Load Model"):
        model = load_model(CONFIG["model_save_path"])
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
        save_trades_to_csv(env.trades)
        st.plotly_chart(plot_equity_curve(env.trades, CONFIG["initial_balance"]), use_container_width=True)
        st.plotly_chart(plot_trades(df, env.trades), use_container_width=True)

elif menu == "Live Trading":
    broker_type = st.sidebar.selectbox("Broker", ["Alpaca", "Binance (CCXT)"])
    api_key = st.sidebar.text_input("API Key")
    api_secret = st.sidebar.text_input("API Secret", type="password")
    if broker_type == "Alpaca":
        base_url = st.sidebar.text_input("Base URL", "https://paper-api.alpaca.markets")
        broker = AlpacaBroker(api_key, api_secret, base_url)
    else:
        broker = CCXTBroker("binance", api_key, api_secret)
    if st.sidebar.button("Check Account"):
        st.write(broker.get_account() if broker_type == "Alpaca" else broker.get_balance())
