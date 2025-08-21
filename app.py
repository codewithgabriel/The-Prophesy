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
import json

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
        height: auto;
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
if 'live_trading_active' not in st.session_state:
    st.session_state.live_trading_active = False
if 'broker' not in st.session_state:
    st.session_state.broker = None
if 'model' not in st.session_state:
    st.session_state.model = None

# Sidebar menu
menu = st.sidebar.radio("Navigation Menu", ["Dashboard", "Backtest", "Live Trading", "Model Training"])

symbol = st.sidebar.text_input("Asset Symbol", CONFIG["asset_symbol"], help="Enter the asset symbol (e.g., BTC/USD)")
if symbol:
    CONFIG["asset_symbol"] = symbol

st.sidebar.text("Select start and end dates for data loading:")
start_date = st.sidebar.date_input("Start Date", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.today())

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")
    
# Dashboard view
if menu == "Dashboard":
    st.header("📊 Trading Dashboard")
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Balance", f"${st.session_state.current_balance:,.2f}")
        
        # Balance indicator
        max_balance = max(st.session_state.balance_history) if st.session_state.balance_history else CONFIG["initial_balance"] * 1.5
        min_balance = min(st.session_state.balance_history) if st.session_state.balance_history else CONFIG["initial_balance"] * 0.5
        balance_range = max_balance - min_balance
        balance_position = ((st.session_state.current_balance - min_balance) / balance_range * 100) if balance_range > 0 else 50
        
        st.markdown(f'<div class="balance-indicator"><div class="balance-marker" style="left: {balance_position}%;"></div></div>', unsafe_allow_html=True)
        
    with col2:
        profit_loss = st.session_state.current_balance - CONFIG["initial_balance"]
        pnl_class = "positive-value" if profit_loss >= 0 else "negative-value"
        st.metric("Profit/Loss", f"${profit_loss:,.2f}", delta=f"{profit_loss/CONFIG['initial_balance']*100:.2f}%")
        
    with col3:
        st.metric("Total Trades", str(len(st.session_state.trade_decisions)))
        
    with col4:
        winning_trades = len([t for t in st.session_state.trade_decisions if t.get("realized_pnl", 0) > 0])
        win_rate = (winning_trades / len(st.session_state.trade_decisions) * 100) if st.session_state.trade_decisions else 0
        st.metric("Win Rate", f"{win_rate:.2f}%")
    
    # Display equity curve if available
    if st.session_state.backtest_results:
        networth, trades = st.session_state.backtest_results
        st.plotly_chart(plot_equity_curve(networth, CONFIG["initial_balance"]), use_container_width=True)
    
    # Recent trade decisions table
    if st.session_state.trade_decisions:
        st.subheader("Recent Trading Decisions")
        recent_trades = pd.DataFrame(st.session_state.trade_decisions[-100:])  # Show last 100 trades
        
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
    
    if st.button("Backtest model"):
        try:
            model = load_model()
            train_df, test_df = load_and_prepare_data(start_date=start_date, end_date=end_date, split=False)
            _, eval_env = create_env(train_df, test_df)
            
            # Initialize progress for backtest
            backtest_progress = st.progress(0)
            backtest_status = st.empty()
            
            # Run backtest with progress updates
            networth, trades = run_backtest(model, test_df, env=eval_env)
            
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
    
    max_trades = st.sidebar.slider("Max trades to display", min_value=50, max_value=5000, value=200, step=1)
    
    # Display backtest results if available
    if st.session_state.backtest_results:
        networth, trades = st.session_state.backtest_results
        _, test_df = load_and_prepare_data(start_date=start_date, end_date=end_date, split=False)

        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Equity Curve", "Trade Analysis", "Performance Metrics"])

        with tab1:
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

                # Use realized_pnl if available, else 0
                winning_trades = len([t for t in trades if t.get("realized_pnl", 0) > 0])
                win_rate = (winning_trades / len(trades) * 100) if trades else 0
                st.metric("Winning Trades", f"{winning_trades} ({win_rate:.2f}%)")

                # Drawdown relative to peak
                running_max = np.maximum.accumulate(networth)
                drawdowns = (running_max - networth) / running_max
                max_dd = np.max(drawdowns) * 100 if len(drawdowns) > 0 else 0
                st.metric("Max Drawdown", f"{max_dd:.2f}%")

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
                st.session_state.broker = broker
                st.sidebar.success("Connected successfully!")
                st.sidebar.json(account_info)
            except Exception as e:
                st.sidebar.error(f"Connection failed: {str(e)}")
    else:
        if st.sidebar.button("Connect to Binance"):
            try:
                broker = CCXTBroker("binance", api_key, api_secret)
                balance = broker.get_balance()
                st.session_state.broker = broker
                st.sidebar.success("Connected successfully!")
                st.sidebar.json(balance)
            except Exception as e:
                st.sidebar.error(f"Connection failed: {str(e)}")
    
    # Live trading controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Trading Controls")
    
    # Load model for live trading
    if st.sidebar.button("Load Model"):
        try:
            model = load_model()
            st.session_state.model = model
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
    
    if st.sidebar.button("Start Live Trading"):
        if st.session_state.broker is None:
            st.error("Please connect to a broker first!")
        elif st.session_state.model is None:
            st.error("Please load the model first!")
        else:
            st.session_state.live_trading_active = True
            st.info("Live trading started. Monitoring market...")
            
            # Start live trading in a separate thread (simplified version)
            try:
                # Get current market data
                symbol = CONFIG["asset_symbol"].replace("/", "") if "/" in CONFIG["asset_symbol"] else CONFIG["asset_symbol"]
                current_price = st.session_state.broker.get_current_price(symbol)
                
                # Use model to make trading decision
                # This is a simplified version - in practice you'd need to format the data
                # to match your model's input requirements
                st.info(f"Current price: ${current_price}")
                
                # For demonstration, we'll simulate a decision
                decision = np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.3, 0.3, 0.4])
                st.success(f"Model decision: {decision}")
                
                if decision == 'BUY':
                    # Execute buy order (simplified)
                    order = st.session_state.broker.place_order(
                        symbol=symbol,
                        quantity=1,  # Example quantity
                        side='buy',
                        order_type='market'
                    )
                    st.session_state.trade_decisions.append({
                        'timestamp': datetime.now(),
                        'action': 'BUY',
                        'price': current_price,
                        'quantity': 1
                    })
                    
                elif decision == 'SELL':
                    # Execute sell order (simplified)
                    order = st.session_state.broker.place_order(
                        symbol=symbol,
                        quantity=1,  # Example quantity
                        side='sell',
                        order_type='market'
                    )
                    st.session_state.trade_decisions.append({
                        'timestamp': datetime.now(),
                        'action': 'SELL',
                        'price': current_price,
                        'quantity': 1
                    })
                
            except Exception as e:
                st.error(f"Trading error: {str(e)}")
        
    if st.sidebar.button("Stop Live Trading"):
        st.session_state.live_trading_active = False
        st.warning("Live trading stopped.")



# Model Training view
elif menu == "Model Training":
    st.header("🤖 Model Training")
    
    st.info("Retrain Model, you can set the env config below")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    if st.button("Start Training"):
        try:
            progress_bar.progress(10)
            train_df, test_df = load_and_prepare_data(start_date=start_date, end_date=end_date)
            status_text.text("Data loaded successfully. Creating environments...")
            progress_bar.progress(30)
            env, eval_env = create_env(train_df, test_df)
            progress_bar.progress(50)
            status_text.text("Environments created successfully. Training model...")
            model = train_ppo_model(env, eval_env)
            st.session_state.model = model
            st.success("Model trained and saved successfully!")
            progress_bar.progress(100)
        except Exception as e:
            st.session_state.training_status = f"Training failed: {str(e)}"
            st.error(f"Training error: {str(e)}")
            progress_bar.empty()
            status_text.empty()

# Configuration
st.sidebar.header("Configuration")
st.sidebar.markdown("Set your trading parameters below:")

# Trading parameters
CONFIG["asset_symbol"] = st.sidebar.text_input("Asset Symbol", CONFIG["asset_symbol"])
CONFIG["timeframe"] = st.sidebar.selectbox("Timeframe", ["1m", "5m", "1h", "1d"])
CONFIG["risk_reward_ratio"] = st.sidebar.slider("Risk/Reward Ratio", 1.0, 5.0, 2.0)
CONFIG["max_drawdown"] = st.sidebar.slider("Max Drawdown", 0.0, 100.0, 20.0)
CONFIG["trading_fee"] = st.sidebar.slider("Trading Fee (%)", 0.0, 5.0, 0.1)

# Model Configuration
CONFIG["csv_path"] = st.sidebar.text_input("CSV Path", CONFIG.get("csv_path", "dataset/aapl.csv"))
CONFIG["start_date"] = st.sidebar.text_input("Start Date", CONFIG.get("start_date", "2015-01-01"))
CONFIG["end_date"] = st.sidebar.text_input("End Date", CONFIG.get("end_date", "2023-01-01"))
CONFIG["train_split"] = st.sidebar.slider("Train Split", 0.5, 1.0, CONFIG.get("train_split", 0.8))
CONFIG["window_size"] = st.sidebar.number_input("Window Size", min_value=1, max_value=500, value=CONFIG.get("window_size", 50))
CONFIG["initial_balance"] = st.sidebar.number_input("Initial Balance", min_value=1.0, value=CONFIG.get("initial_balance", 100.0))
CONFIG["commission_pct"] = st.sidebar.number_input("Commission (%)", min_value=0.0, max_value=0.01, value=CONFIG.get("commission_pct", 0.001))
CONFIG["commission_fixed"] = st.sidebar.number_input("Commission Fixed", min_value=0.0, value=CONFIG.get("commission_fixed", 0.0))
CONFIG["spread_pct"] = st.sidebar.number_input("Spread (%)", min_value=0.0, max_value=0.01, value=CONFIG.get("spread_pct", 0.0001))
CONFIG["slippage_coeff"] = st.sidebar.number_input("Slippage Coeff", min_value=0.0, max_value=0.01, value=CONFIG.get("slippage_coeff", 0.0002))
CONFIG["volume_limit"] = st.sidebar.slider("Volume Limit", 0.0, 1.0, CONFIG.get("volume_limit", 0.1))
CONFIG["max_position_size"] = st.sidebar.number_input("Max Position Size", min_value=1, value=CONFIG.get("max_position_size", 10000))
CONFIG["max_risk_per_trade"] = st.sidebar.slider("Max Risk Per Trade", 0.0, 1.0, CONFIG.get("max_risk_per_trade", 0.02))
CONFIG["stop_loss_pct"] = st.sidebar.slider("Stop Loss (%)", 0.0, 1.0, CONFIG.get("stop_loss_pct", 0.02))
CONFIG["drawdown_scale_threshold"] = st.sidebar.slider("Drawdown Scale Threshold", 0.0, 1.0, CONFIG.get("drawdown_scale_threshold", 0.1))
CONFIG["drawdown_scale_factor"] = st.sidebar.slider("Drawdown Scale Factor", 0.0, 1.0, CONFIG.get("drawdown_scale_factor", 0.5))
CONFIG["volatility_scaling"] = st.sidebar.checkbox("Volatility Scaling", value=CONFIG.get("volatility_scaling", True))
CONFIG["max_leverage"] = st.sidebar.slider("Max Leverage", 1.0, 10.0, CONFIG.get("max_leverage", 2.0))
CONFIG["maintenance_margin"] = st.sidebar.slider("Maintenance Margin", 0.0, 1.0, CONFIG.get("maintenance_margin", 0.25))
CONFIG["financing_rate_annual"] = st.sidebar.slider("Financing Rate Annual", 0.0, 1.0, CONFIG.get("financing_rate_annual", 0.02))
CONFIG["reward_scaling"] = st.sidebar.slider("Reward Scaling", 0.0, 10.0, CONFIG.get("reward_scaling", 1.0))
CONFIG["dd_penalty_coeff"] = st.sidebar.slider("Drawdown Penalty Coeff", 0.0, 10.0, CONFIG.get("dd_penalty_coeff", 0.0))
CONFIG["turnover_penalty_coeff"] = st.sidebar.slider("Turnover Penalty Coeff", 0.0, 10.0, CONFIG.get("turnover_penalty_coeff", 0.0))
CONFIG["normalize_observations"] = st.sidebar.checkbox("Normalize Observations", value=CONFIG.get("normalize_observations", True))
CONFIG["random_start"] = st.sidebar.checkbox("Random Start", value=CONFIG.get("random_start", True))
CONFIG["episode_length"] = st.sidebar.number_input("Episode Length", min_value=0, value=CONFIG.get("episode_length", 0) or 0)
CONFIG["total_timesteps"] = st.sidebar.number_input("Total Timesteps", min_value=1, value=CONFIG.get("total_timesteps", 300000))
CONFIG["tensorboard_log_dir"] = st.sidebar.text_input("Tensorboard Log Dir", CONFIG.get("tensorboard_log_dir", "saved_models/ppo/tb_logs"))
CONFIG["model_save_path"] = st.sidebar.text_input("Model Save Path", CONFIG.get("model_save_path", "saved_models/ppo_trader_model/final_model.zip"))

# Save configuration
if st.sidebar.button("Save Config"):
    with open("config.json", "w") as f:
        json.dump(CONFIG, f)
    st.sidebar.success("Configuration saved successfully!")