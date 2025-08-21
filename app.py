import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from utils import load_model, load_and_prepare_data, create_env, run_backtest
from config import CONFIG

st.set_page_config(page_title="RL Trading Dashboard", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Backtest", "Performance Metrics"])

# --- Session State Initialization ---
if "balance_history" not in st.session_state:
    st.session_state.balance_history = [CONFIG["initial_balance"]]
if "current_balance" not in st.session_state:
    st.session_state.current_balance = CONFIG["initial_balance"]
if "trade_decisions" not in st.session_state:
    st.session_state.trade_decisions = []
if "backtest_results" not in st.session_state:
    st.session_state.backtest_results = None

# --- Dashboard Page ---
if page == "Dashboard":
    st.title("📊 RL Trading Dashboard")
    st.metric("💰 Current Balance", f"${st.session_state.current_balance:,.2f}")

    st.subheader("Balance Over Time")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(st.session_state.balance_history, label="Portfolio Value", color="blue")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Balance ($)")
    ax.legend()
    st.pyplot(fig)

    if st.session_state.trade_decisions:
        st.subheader("Recent Trading Decisions")
        recent_trades = pd.DataFrame(st.session_state.trade_decisions[-100:])
        display_df = recent_trades.copy()

        # Handle timestamps
        if 'timestamp' in display_df.columns:
            try:
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                pass

        # Determine action
        if 'position_shares' in display_df.columns:
            display_df['action'] = display_df['position_shares'].apply(
                lambda x: 'BUY' if x > 0 else 'SELL' if x < 0 else 'HOLD'
            )
        elif 'side' in display_df.columns:
            display_df['action'] = display_df['side'].str.upper()
        elif 'action' not in display_df.columns:
            display_df['action'] = "HOLD"

        def color_trade_rows(row):
            if row['action'] == 'BUY':
                return ['background-color: rgba(0, 200, 83, 0.1)'] * len(row)
            elif row['action'] == 'SELL':
                return ['background-color: rgba(255, 82, 82, 0.1)'] * len(row)
            return [''] * len(row)

        styled_df = display_df.style.apply(color_trade_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True)

# --- Backtest Page ---
elif page == "Backtest":
    st.title("🔄 Backtest Model")
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime(2021, 1, 1))

    if st.button("Backtest model", use_container_width=True):
        try:
            model = load_model()
            train_df, test_df = load_and_prepare_data(start_date=start_date, end_date=end_date, split=False)
            _, eval_env = create_env(train_df, test_df)

            backtest_result = run_backtest(model, test_df, env=eval_env)

            # Handle dict or tuple outputs
            if isinstance(backtest_result, dict):
                networth = np.array(backtest_result.get("net_worth", []), dtype=float)
                trades = backtest_result.get("trades", [])
            else:
                networth, trades = backtest_result
                networth = np.array(networth, dtype=float)

            st.session_state.backtest_results = (networth, trades)
            st.session_state.trade_decisions = trades
            st.session_state.balance_history = networth.tolist()
            st.session_state.current_balance = float(networth[-1]) if len(networth) > 0 else CONFIG["initial_balance"]

            st.success("✅ Backtest completed successfully!")
            st.rerun()

        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")

# --- Performance Metrics Page ---
elif page == "Performance Metrics":
    st.title("📈 Performance Metrics")

    if st.session_state.backtest_results is None:
        st.warning("⚠️ Run a backtest first!")
    else:
        networth, trades = st.session_state.backtest_results

        st.subheader("Portfolio Growth")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(networth, label="Portfolio Value", color="blue")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Net Worth ($)")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Performance Summary")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Final Portfolio Value", f"${networth[-1]:,.2f}")
            total_return = (networth[-1] - networth[0]) / networth[0] * 100
            st.metric("Total Return", f"{total_return:.2f}%")

        with col2:
            st.metric("Number of Trades", len(trades))
            winning_trades = len([t for t in trades if t.get('profit', 0) > 0])
            win_rate = (winning_trades/len(trades)*100) if trades else 0
            st.metric("Winning Trades", f"{winning_trades} ({win_rate:.2f}%)")

            if len(networth) > 1:
                running_max = np.maximum.accumulate(networth)
                drawdowns = (running_max - networth) / running_max
                max_dd = np.max(drawdowns) * 100
            else:
                max_dd = 0.0
            st.metric("Max Drawdown", f"{max_dd:.2f}%")

        with col3:
            daily_returns = pd.Series(networth).pct_change().dropna()
            sharpe_ratio = (
                np.sqrt(252) * daily_returns.mean() / daily_returns.std()
                if not daily_returns.empty and daily_returns.std() != 0 else 0
            )
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

        st.subheader("Trade Log")
        if trades:
            trades_df = pd.DataFrame(trades)

            if 'timestamp' in trades_df.columns:
                try:
                    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    pass

            st.dataframe(trades_df, use_container_width=True)

        else:
            st.info("No trades were executed during backtest.")
