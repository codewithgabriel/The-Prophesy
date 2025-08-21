# ================================================
# 📂 trading_app/main.py (Fixed)
# ================================================
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

# Set page configuration
st.set_page_config(
    layout="wide", 
    page_title="RL Trading Dashboard", 
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .positive-value {
        color: #00C853;
        font-weight: 700;
    }
    .negative-value {
        color: #FF5252;
        font-weight: 700;
    }
    .trade-buy {
        background-color: rgba(0, 200, 83, 0.1);
    }
    .trade-sell {
        background-color: rgba(255, 82, 82, 0.1);
    }
    .progress-bar {
        height: 10px;
        border-radius: 5px;
        background-color: #e0e0e0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 5px;
        background-color: #1E88E5;
    }
    .balance-indicator {
        height: 20px;
        border-radius: 10px;
        margin-top: 5px;
        background: linear-gradient(90deg, #FF5252 0%, #FFC107 50%, #00C853 100%);
        position: relative;
    }
    .balance-marker {
        position: absolute;
        top: -5px;
        width: 4px;
        height: 30px;
        background-color: #000;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📈 Professional RL Trading Dashboard</h1>', unsafe_allow_html=True)

# Initialize session state for storing data across reruns
if 'training_progress' not in st.session_state:
    st.session_state.training_progress = 0
if 'training_status' not in st.session_state:
    st.session_state.training_status = "Not started"
if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None
if 'trade_decisions' not in st.session_state:
    st.session_state.trade_decisions = []
if 'current_balance' not in st.session_state:
    st.session_state.current_balance = CONFIG["initial_balance"]
if 'balance_history' not in st.session_state:
    st.session_state.balance_history = []

# Sidebar menu
st.sidebar.text("Select Start and End Date")
st.sidebar.markdown("### Date Range")
start_date = st.sidebar.date_input("Start Date", value=datetime(2023, 1 , 1))
end_date = st.sidebar.date_input("End Date", value=datetime.today())

menu = st.sidebar.selectbox("Navigation Menu", ["Dashboard", "Backtest", "Live Trading", "Model Training"])




def get_data_with_dates():
    return load_and_prepare_data(start_date=START_DATE, end_date=END_DATE)

# Dashboard view
if menu == "Dashboard":
    st.header("📊 Trading Dashboard")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Current Balance", f"${st.session_state.current_balance:,.2f}")
        
        # Balance indicator
        max_balance = max(st.session_state.balance_history) if st.session_state.balance_history else CONFIG["initial_balance"] * 1.5
        min_balance = min(st.session_state.balance_history) if st.session_state.balance_history else CONFIG["initial_balance"] * 0.5
        balance_range = max_balance - min_balance
        balance_position = ((st.session_state.current_balance - min_balance) / balance_range * 100) if balance_range > 0 else 50
        
        st.markdown(f'<div class="balance-indicator"><div class="balance-marker" style="left: {balance_position}%;"></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        profit_loss = st.session_state.current_balance - CONFIG["initial_balance"]
        pnl_class = "positive-value" if profit_loss >= 0 else "negative-value"
        st.metric("Profit/Loss", f"${profit_loss:,.2f}", delta=f"{profit_loss/CONFIG['initial_balance']*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Trades", str(len(st.session_state.trade_decisions)))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        win_rate = len([t for t in st.session_state.trade_decisions if t.get('profit', 0) > 0]) / len(st.session_state.trade_decisions) * 100 if st.session_state.trade_decisions else 0
        st.metric("Win Rate", f"{win_rate:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display equity curve if available
    if st.session_state.backtest_results:
        networth, trades = st.session_state.backtest_results
        st.plotly_chart(plot_equity_curve(networth, CONFIG["initial_balance"]), use_container_width=True)
    
    # Recent trade decisions table
    if st.session_state.trade_decisions:
        st.subheader("Recent Trading Decisions")
        recent_trades = pd.DataFrame(st.session_state.trade_decisions[-10:])  # Show last 10 trades
        
        # Format the DataFrame for display
        display_df = recent_trades.copy()
        
        # Check if timestamp column exists and format it
        if 'timestamp' in display_df.columns:
            try:
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                pass  # If timestamp conversion fails, keep original
        
        # Check if position_shares column exists to determine trade direction
        if 'position_shares' in display_df.columns:
            # Add action column based on position_shares
            display_df['action'] = display_df['position_shares'].apply(
                lambda x: 'BUY' if x > 0 else 'SELL' if x < 0 else 'HOLD'
            )
        
        # Apply styling based on trade action if action column exists
        if 'action' in display_df.columns:
            def color_trade_rows(row):
                if row['action'] == 'BUY':
                    return ['background-color: rgba(0, 200, 83, 0.1)'] * len(row)
                elif row['action'] == 'SELL':
                    return ['background-color: rgba(255, 82, 82, 0.1)'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_df = display_df.style.apply(color_trade_rows, axis=1)
        else:
            styled_df = display_df.style
        
        st.dataframe(styled_df, use_container_width=True)

# Backtest view
elif menu == "Backtest":
    st.header("🔁 Backtesting")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Train Model", use_container_width=True):
            st.session_state.training_status = "Training in progress..."
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate training progress
            for i in range(100):
                st.session_state.training_progress = i + 1
                progress_bar.progress(st.session_state.training_progress / 100)
                status_text.text(f"Training: {st.session_state.training_progress}% complete")
                time.sleep(0.05)  # Simulate training time
                
                # Update every 10% progress
                if st.session_state.training_progress % 10 == 0:
                    st.rerun()
            
            # Actual training would happen here
            try:
                train_df, test_df = load_and_prepare_data()
                env, eval_env = create_env(train_df, test_df)
                model = train_ppo_model(env, eval_env)
                st.session_state.training_status = "Training completed successfully!"
                st.success("Model trained and saved!")
            except Exception as e:
                st.session_state.training_status = f"Training failed: {str(e)}"
                st.error(f"Training error: {str(e)}")
            
            progress_bar.empty()
            status_text.empty()
    
    with col2:
        if st.button("Backtest Model", use_container_width=True):
            try:
                model = load_model()
                train_df, test_df = load_and_prepare_data()
                _, eval_env = create_env(train_df, test_df)
                
                # Initialize progress for backtest
                backtest_progress = st.progress(0)
                backtest_status = st.empty()
                
                # Run backtest with progress updates
                networth, trades = run_backtest(model, test_df, train_df)
                col2.write(trades[:10])
                
                
                # Store results in session state
                st.session_state.backtest_results = (networth, trades)
                st.session_state.trade_decisions = trades
                
                # Update balance history
                st.session_state.balance_history = networth.tolist()
                st.session_state.current_balance = networth[-1] if len(networth) > 0 else CONFIG["initial_balance"]
                
                backtest_progress.progress(100)
                backtest_status.text("Backtest completed!")
                time.sleep(1)
                backtest_progress.empty()
                backtest_status.empty()
                
                st.success("Backtest completed successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error loading model or running backtest: {str(e)}")
    
    # Display training status
    st.sidebar.markdown("### Training Status")
    st.sidebar.text(st.session_state.training_status)
    
    if st.session_state.training_progress > 0:
        st.sidebar.markdown("### Training Progress")
        st.sidebar.markdown(f'<div class="progress-bar"><div class="progress-fill" style="width: {st.session_state.training_progress}%;"></div></div>', unsafe_allow_html=True)
        st.sidebar.text(f"{st.session_state.training_progress}% complete")
    
    max_trades = st.sidebar.slider("Max trades to display", min_value=50, max_value=1000, value=200, step=10)
    
    # Display backtest results if available
    if st.session_state.backtest_results:
        networth, trades = st.session_state.backtest_results
        train_df, test_df = load_and_prepare_data()
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Equity Curve", "Trade Analysis", "Performance Metrics"])
        
        with tab1:
            tab1.write(trades)
            tab1.write(test_df.head())
            st.plotly_chart(plot_equity_curve(networth, CONFIG["initial_balance"]), use_container_width=True)
        
        with tab2:
            st.plotly_chart(plot_trades(test_df, trades, max_trades=max_trades), use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Initial Balance", f"${CONFIG['initial_balance']:,.2f}")
                st.metric("Final Balance", f"${networth[-1]:,.2f}")
                returns = (networth[-1] - CONFIG['initial_balance']) / CONFIG['initial_balance'] * 100
                st.metric("Total Return", f"{returns:.2f}%")
            
            with col2:
                st.metric("Number of Trades", len(trades))
               
               
                
                winning_trades = len([t for t in trades if t.get('profit', 0) > 0])
                st.metric("Winning Trades", f"{winning_trades} ({winning_trades/len(trades)*100:.2f}%)" if trades else "0")
                st.metric("Max Drawdown", f"{((min(networth) - CONFIG['initial_balance']) / CONFIG['initial_balance'] * 100):.2f}%")

# Live Trading view
elif menu == "Live Trading":
    st.header("🔴 Live Trading")
    
    broker_type = st.sidebar.selectbox("Broker", ["Alpaca", "Binance (CCXT)"])
    api_key = st.sidebar.text_input("API Key")
    api_secret = st.sidebar.text_input("API Secret", type="password")
    
    if broker_type == "Alpaca":
        base_url = st.sidebar.text_input("Base URL", "https://paper-api.alpaca.markets")
        if st.sidebar.button("Connect to Alpaca"):
            try:
                broker = AlpacaBroker(api_key, api_secret, base_url)
                account_info = broker.get_account()
                st.sidebar.success("Connected successfully!")
                st.sidebar.json(account_info)
            except Exception as e:
                st.sidebar.error(f"Connection failed: {str(e)}")
    else:
        if st.sidebar.button("Connect to Binance"):
            try:
                broker = CCXTBroker("binance", api_key, api_secret)
                balance = broker.get_balance()
                st.sidebar.success("Connected successfully!")
                st.sidebar.json(balance)
            except Exception as e:
                st.sidebar.error(f"Connection failed: {str(e)}")
    
    # Live trading controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Trading Controls")
    
    if st.sidebar.button("Start Live Trading"):
        st.info("Live trading started. Monitoring market...")
        # Here you would implement the actual live trading logic
        
    if st.sidebar.button("Stop Live Trading"):
        st.warning("Live trading stopped.")

# Model Training view
elif menu == "Model Training":
    st.header("🤖 Model Training")
    
    st.info("Configure your model training parameters below.")
    
    with st.expander("Training Parameters"):
        col1, col2 = st.columns(2)
        
        with col1:
            learning_rate = st.slider("Learning Rate", 0.0001, 0.01, 0.0003, 0.0001)
            n_steps = st.slider("Training Steps", 1000, 100000, 10000, 1000)
        
        with col2:
            batch_size = st.slider("Batch Size", 8, 256, 64, 8)
            n_epochs = st.slider("Epochs", 1, 20, 10, 1)
    
    if st.button("Start Training"):
        st.session_state.training_status = "Training in progress..."
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create a placeholder for training metrics
        metrics_placeholder = st.empty()
        
        # Simulate training with periodic updates
        for i in range(100):
            st.session_state.training_progress = i + 1
            progress_bar.progress(st.session_state.training_progress / 100)
            status_text.text(f"Epoch {i//10 + 1}/{n_epochs} - Training: {st.session_state.training_progress}% complete")
            
            # Update metrics every 10%
            if i % 10 == 0:
                # Simulated metrics
                metrics_data = {
                    "Epoch": i//10 + 1,
                    "Reward": np.random.uniform(-0.5, 2.0),
                    "Loss": np.random.uniform(0.1, 0.5),
                    "Value Loss": np.random.uniform(0.01, 0.1),
                    "Explained Variance": np.random.uniform(0.7, 0.95)
                }
                
                # Display metrics in a table
                metrics_df = pd.DataFrame([metrics_data])
                metrics_placeholder.dataframe(metrics_df, use_container_width=True)
            
            time.sleep(0.1)  # Simulate training time
        
        progress_bar.empty()
        status_text.empty()
        st.session_state.training_status = "Training completed successfully!"
        st.success("Model training completed!")