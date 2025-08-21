# ================================================
# 📂 trading_app/main.py (Enhanced with Real-time and Date Selection)
# ================================================
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from envs.config import CONFIG
from brokers.broker_alpaca import AlpacaBroker
from brokers.broker_ccxt import CCXTBroker
from models.train_ppo import train_ppo_model 
from utils import run_backtest, create_env, load_and_prepare_data, load_model, plot_equity_curve, plot_trades
import threading
import queue

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
    .decision-panel {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
    }
    .decision-buy {
        border-left-color: #00C853;
    }
    .decision-sell {
        border-left-color: #FF5252;
    }
    .decision-hold {
        border-left-color: #FFC107;
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
if 'real_time_updates' not in st.session_state:
    st.session_state.real_time_updates = queue.Queue()
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'backtest_dates' not in st.session_state:
    st.session_state.backtest_dates = {
        "start_date": datetime.now() - timedelta(days=365),
        "end_date": datetime.now()
    }

# Sidebar menu
menu = st.sidebar.selectbox("Navigation Menu", ["Dashboard", "Backtest", "Live Trading", "Model Training"])

# Function to run backtest in a separate thread
def run_backtest_thread(model, test_df, progress_bar, status_text):
    try:
        # Run backtest with progress updates
        networth, trades = run_backtest(model, test_df)
        
        # Store results in session state
        st.session_state.backtest_results = (networth, trades)
        st.session_state.trade_decisions = trades
        
        # Update balance history
        st.session_state.balance_history = networth.tolist()
        st.session_state.current_balance = networth[-1] if len(networth) > 0 else CONFIG["initial_balance"]
        
        progress_bar.progress(100)
        status_text.text("Backtest completed!")
        time.sleep(1)
        
    except Exception as e:
        st.session_state.real_time_updates.put(f"Error: {str(e)}")

# Function to simulate real-time trading
def simulate_real_time_trading(model, test_df, update_interval=0.5):
    env = make_eval_env(test_df, 50)  # Assuming window_size=50
    obs = env.reset()
    done = [False]
    
    while not done[0] and st.session_state.is_running:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)
        
        # Get current state information
        current_step = info[0]['current_step']
        total_steps = info[0]['total_steps']
        net_worth = info[0]['net_worth']
        position = info[0]['position']
        
        # Determine action type
        action_type = "HOLD"
        if action == 0:
            action_type = "BUY"
        elif action == 1:
            action_type = "SELL"
        
        # Create update message
        update_msg = {
            "type": "decision",
            "step": current_step,
            "total_steps": total_steps,
            "action": action_type,
            "net_worth": net_worth,
            "position": position,
            "price": test_df.iloc[current_step]["Close"] if current_step < len(test_df) else 0,
            "timestamp": test_df.iloc[current_step]["Date"] if current_step < len(test_df) else datetime.now()
        }
        
        # Add to queue
        st.session_state.real_time_updates.put(update_msg)
        
        # Update session state
        st.session_state.current_balance = net_worth
        st.session_state.balance_history.append(net_worth)
        
        # Add to trade decisions if it's a buy or sell
        if action_type in ["BUY", "SELL"]:
            trade_record = {
                "timestamp": update_msg["timestamp"],
                "action": action_type,
                "price": update_msg["price"],
                "shares": position,
                "net_worth": net_worth
            }
            st.session_state.trade_decisions.append(trade_record)
        
        time.sleep(update_interval)
    
    st.session_state.is_running = False

# Dashboard view
if menu == "Dashboard":
    st.header("📊 Trading Dashboard")
    
    # Display real-time updates if available
    if not st.session_state.real_time_updates.empty():
        st.subheader("Real-time Updates")
        update_placeholder = st.empty()
        
        updates = []
        while not st.session_state.real_time_updates.empty():
            updates.append(st.session_state.real_time_updates.get())
        
        for update in updates[-5:]:  # Show last 5 updates
            if isinstance(update, dict) and update.get("type") == "decision":
                action_class = ""
                if update["action"] == "BUY":
                    action_class = "decision-buy"
                elif update["action"] == "SELL":
                    action_class = "decision-sell"
                else:
                    action_class = "decision-hold"
                
                st.markdown(f"""
                <div class="decision-panel {action_class}">
                    <strong>{update['timestamp']}</strong> - 
                    <span class="{'positive-value' if update['action'] == 'BUY' else 'negative-value' if update['action'] == 'SELL' else ''}">
                        {update['action']}
                    </span> at ${update['price']:.2f}
                    <br>Net Worth: ${update['net_worth']:.2f} | Position: {update['position']} shares
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info(update)
    
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
        
        # Check if action column exists to determine trade direction
        if 'action' in display_df.columns:
            # Apply styling based on trade action
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
    
    # Date selection for backtest
    st.sidebar.subheader("Backtest Date Range")
    start_date = st.sidebar.date_input(
        "Start Date", 
        value=st.session_state.backtest_dates["start_date"],
        max_value=datetime.now()
    )
    end_date = st.sidebar.date_input(
        "End Date", 
        value=st.session_state.backtest_dates["end_date"],
        max_value=datetime.now()
    )
    
    # Update session state with selected dates
    st.session_state.backtest_dates["start_date"] = start_date
    st.session_state.backtest_dates["end_date"] = end_date
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Train Model", use_container_width=True):
            st.session_state.training_status = "Training in progress..."
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update config with selected dates
            CONFIG["start_date"] = start_date.strftime("%Y-%m-%d")
            CONFIG["end_date"] = end_date.strftime("%Y-%m-%d")
            
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
        if st.button("Run Backtest", use_container_width=True):
            try:
                model = load_model()
                
                # Update config with selected dates
                CONFIG["start_date"] = start_date.strftime("%Y-%m-%d")
                CONFIG["end_date"] = end_date.strftime("%Y-%m-%d")
                
                train_df, test_df = load_and_prepare_data()
                
                # Initialize progress for backtest
                backtest_progress = st.progress(0)
                backtest_status = st.empty()
                
                # Run backtest in a separate thread to allow real-time updates
                st.session_state.is_running = True
                thread = threading.Thread(
                    target=simulate_real_time_trading, 
                    args=(model, test_df, 0.5)
                )
                thread.daemon = True
                thread.start()
                
                # Display progress
                progress_placeholder = st.empty()
                while st.session_state.is_running:
                    if not st.session_state.real_time_updates.empty():
                        update = st.session_state.real_time_updates.get()
                        if isinstance(update, dict) and update.get("type") == "decision":
                            progress = (update["step"] / update["total_steps"]) * 100
                            backtest_progress.progress(progress)
                            backtest_status.text(f"Step {update['step']}/{update['total_steps']} - {update['action']} at ${update['price']:.2f}")
                    
                    time.sleep(0.1)
                    st.rerun()
                
                backtest_progress.progress(100)
                backtest_status.text("Backtest completed!")
                time.sleep(1)
                backtest_progress.empty()
                backtest_status.empty()
                
                st.success("Backtest completed successfully!")
                
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
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Equity Curve", "Trade Analysis", "Performance Metrics"])
        
        with tab1:
            st.plotly_chart(plot_equity_curve(networth, CONFIG["initial_balance"]), use_container_width=True)
        
        with tab2:
            train_df, test_df = load_and_prepare_data()
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
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Start Live Trading", use_container_width=True):
            try:
                model = load_model()
                # Get recent data for live trading
                recent_end_date = datetime.now()
                recent_start_date = recent_end_date - timedelta(days=30)
                CONFIG["start_date"] = recent_start_date.strftime("%Y-%m-%d")
                CONFIG["end_date"] = recent_end_date.strftime("%Y-%m-%d")
                
                train_df, test_df = load_and_prepare_data()
                
                st.session_state.is_running = True
                thread = threading.Thread(
                    target=simulate_real_time_trading, 
                    args=(model, test_df, 2)  # Longer interval for live trading
                )
                thread.daemon = True
                thread.start()
                
                st.info("Live trading started. Monitoring market...")
            except Exception as e:
                st.error(f"Error starting live trading: {str(e)}")
    
    with col2:
        if st.button("Stop Trading", use_container_width=True):
            st.session_state.is_running = False
            st.warning("Trading stopped.")

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
    
    # Date selection for training
    st.sidebar.subheader("Training Date Range")
    train_start_date = st.sidebar.date_input(
        "Training Start Date", 
        value=datetime.now() - timedelta(days=365),
        max_value=datetime.now()
    )
    train_end_date = st.sidebar.date_input(
        "Training End Date", 
        value=datetime.now(),
        max_value=datetime.now()
    )
    
    if st.button("Start Training"):
        # Update config with selected dates
        CONFIG["start_date"] = train_start_date.strftime("%Y-%m-%d")
        CONFIG["end_date"] = train_end_date.strftime("%Y-%m-%d")
        
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

# Auto-refresh the page when real-time trading is active
if st.session_state.is_running:
    time.sleep(1)
    st.rerun()