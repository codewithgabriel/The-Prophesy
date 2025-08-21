# ================================================
# 📂 utils.py (Enhanced)
# ================================================
import pandas as pd
from envs.config import CONFIG
from technical_analysis import add_technical_indicators
import os
from data_fetcher import download_price_data
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from envs.make_env import make_env
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_model():
    return PPO.load(CONFIG['model_save_path'])

def save_trades_to_csv(trades, filename="trades.csv"):
    pd.DataFrame(trades).to_csv(filename, index=False)

def run_backtest(model, test_df, env):
    """
    Run backtest using trained RL model.
    Returns:
        equity_curve: np.ndarray of net worth values
        trades: trade log (copied during execution)
    """
   
    obs = env.reset()
    net_worths = []
    trades = []  # We'll store copies here during execution
    done = [False]

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Store current net worth
        net_worths.append(info[0]['net_worth'])
        
        # Make a COPY of current trades and extend our maintained list
        current_trades = getattr(env.envs[0], 'trades', [])
        if current_trades:
            trades.extend(current_trades.copy())  # Explicit copy
        
    return np.array(net_worths), trades

def load_and_prepare_data(start_date, end_date, split=True):
    # # 1) Load & prepare data
    # if CONFIG["csv_path"] and os.path.exists(CONFIG["csv_path"]):
    #     print("Loading data from CSV...")
    #     df = pd.read_csv(CONFIG["csv_path"])
    #     df["Date"] = pd.to_datetime(df["Date"])
    #     print(df.columns.tolist())
    # else:
    #     print("Downloading data from source...")
    
    df = download_price_data(CONFIG["asset_symbol"], start_date, end_date)
    df["Date"] = pd.to_datetime(df["Date"])
    print(df.columns.tolist())

    df = df.sort_values("Date").reset_index(drop=True)
    print(df.columns.tolist())
    # Ensure numeric types for OHLCV data
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # convert to float, NaN if invalid
    df = add_technical_indicators(df)

    # 2) Train/test split
    if split:
        split_idx = int(len(df) * CONFIG["train_split"])
        train_df = df.iloc[:split_idx].reset_index(drop=True)
        test_df = df.iloc[split_idx:].reset_index(drop=True)
        return train_df, test_df
    else:
        return df, df

def create_env(train_df, test_df):
     # 3) Vectorized environments for parallelism
    env = DummyVecEnv([lambda: make_env(train_df, train_mode=True)])
    eval_env = DummyVecEnv([lambda: make_env(test_df, train_mode=False)])
    return env, eval_env

# Enhanced visualization functions
def plot_equity_curve(networth, initial_balance):
    # Calculate daily returns
    returns = np.diff(networth) / networth[:-1] * 100
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Equity Curve', 'Daily Returns'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Equity curve
    fig.add_trace(
        go.Scatter(
            y=networth, 
            mode="lines", 
            name="Equity",
            line=dict(color='#1E88E5', width=2),
            fill='tozeroy',
            fillcolor='rgba(30, 136, 229, 0.1)'
        ),
        row=1, col=1
    )
    
    # Add initial balance reference line
    fig.add_hline(
        y=initial_balance, 
        line_dash="dash", 
        line_color="green", 
        annotation_text="Initial Balance",
        row=1, col=1
    )
    
    # Daily returns
    colors = ['green' if r >= 0 else 'red' for r in returns]
    fig.add_trace(
        go.Bar(
            y=returns,
            name='Returns',
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    fig.update_yaxes(title_text="Balance ($)", row=1, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    
    return fig

def plot_trades(df, trades, max_trades=100):
    # Create subplots with volume
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Price with trades', 'Volume'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True
    )
    
    # Price chart
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
            increasing_line_color='#2E7D32',
            decreasing_line_color='#D32F2F'
        ),
        row=1, col=1
    )
    
    # Limit number of trades to render
    trades_to_plot = trades[-max_trades:] if max_trades < len(trades) else trades
    
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []
    
    for t in trades_to_plot:
        trade_time = df.loc[t["index"], "Date"]
        trade_price = df.loc[t["index"], "Close"]
        
        if t["position_shares"] > 0:  # Buy
            buy_dates.append(trade_time)
            buy_prices.append(trade_price)
        else:  # Sell
            sell_dates.append(trade_time)
            sell_prices.append(trade_price)
    
    # Add buy markers
    if buy_dates:
        fig.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode="markers",
                marker=dict(color="#00C853", size=10, symbol="triangle-up", line=dict(width=2, color="DarkGreen")),
                name="Buy",
                hovertemplate="Buy: %{y:.2f}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Add sell markers
    if sell_dates:
        fig.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode="markers",
                marker=dict(color="#FF5252", size=10, symbol="triangle-down", line=dict(width=2, color="DarkRed")),
                name="Sell",
                hovertemplate="Sell: %{y:.2f}<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            name="Volume",
            marker_color='#546E7A',
            opacity=0.5
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#2c3e50'),
        margin=dict(l=50, r=50, t=50, b=50),
        xaxis_rangeslider_visible=False
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    return fig