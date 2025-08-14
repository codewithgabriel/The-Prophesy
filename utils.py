# ================================================
# 📂 trading_app/utils.py
# ================================================
import pandas as pd
from envs.config import CONFIG
from technical_analysis import add_technical_indicators
import os
from data_fetcher import download_price_data
import numpy as np
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from envs.make_env import make_env, make_eval_env

def load_model():
    return PPO.load(CONFIG['model_save_path'])

def save_trades_to_csv(trades, filename="trades.csv"):
    pd.DataFrame(trades).to_csv(filename, index=False)


def run_backtest(model , test_df, window_size=50):
    """
    Run backtest using trained RL model.
    Returns:
        equity_curve: np.ndarray of net worth values
        trades: trade log (copied during execution)
    """
    env = make_eval_env(test_df, window_size)
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


def load_and_prepare_data():
# 1) Load & prepare data
    if CONFIG["csv_path"] and os.path.exists(CONFIG["csv_path"]):
        df = pd.read_csv(CONFIG["csv_path"])
    else:
        df = download_price_data(CONFIG["asset_symbol"], CONFIG["start_date"], CONFIG["end_date"])
    
    # Ensure numeric types for OHLCV data
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # convert to float, NaN if invalid
    df = add_technical_indicators(df)

    # 2) Train/test split
    split_idx = int(len(df) * CONFIG["train_split"])
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    return train_df, test_df

def create_env(train_df, test_df):
     # 3) Vectorized environments for parallelism
    env = DummyVecEnv([lambda: make_env(train_df, train_mode=True)])
    eval_env = DummyVecEnv([lambda: make_env(test_df, train_mode=False)])
    return env, eval_env