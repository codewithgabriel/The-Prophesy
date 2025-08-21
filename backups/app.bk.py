
import streamlit as st
import pandas as pd
from envs.config import CONFIG
from brokers.broker_alpaca import AlpacaBroker
from brokers.broker_ccxt import CCXTBroker
from models.train_ppo import  train_ppo_model 
from utils import run_backtest , create_env , load_and_prepare_data, load_model , plot_equity_curve, plot_trades


st.set_page_config(layout="wide")
st.title("📈 RL Trading Dashboard")

menu = st.sidebar.selectbox("Menu", ["Backtest", "Live Trading"])

if menu == "Backtest":
    if st.sidebar.button("Train Model"):
        train_df, test_df = load_and_prepare_data()
        env, eval_env = create_env(train_df, test_df)
        model = train_ppo_model(env ,  eval_env)
        st.success("Model trained and saved!")

    max_trades = st.sidebar.slider("Max trades to display", min_value=50, max_value=1000, value=200, step=10)

    if st.sidebar.button("Load Model"):
        model = load_model()
        train_df, test_df = load_and_prepare_data()
        _, eval_env = create_env(train_df, test_df)
        networth, trades = run_backtest(model, test_df)
        st.plotly_chart(plot_equity_curve(networth, CONFIG["initial_balance"]), use_container_width=True)
        st.plotly_chart(plot_trades(test_df, trades, max_trades=max_trades), use_container_width=True)

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
