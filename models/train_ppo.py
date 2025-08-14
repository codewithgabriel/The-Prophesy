"""
train.py
---------
Training script for RL-based trading agent using PPO with a realistic market simulation.
Includes:
 - Configurable training/testing split
 - TensorBoard logging
 - Evaluation callback with early stopping
 - Modular design for easy extensions
"""

import os
import pandas as pd
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from envs.trading_env import TradingEnv
from technical_analysis import add_technical_indicators
from envs.config import CONFIG
# Custom imports (replace with your paths)
# from features import add_technical_indicators
# from envs.realistic_trading_env import RealisticTradingEnv  # our upgraded env

# ========================
# ENV FACTORY
# ========================
def make_env(df, train_mode=True):
    """
    Creates the trading environment instance with consistent parameters.
    """
    feature_cols = ["Close", "ret", "sma_10", "sma_50", "rsi_14", "macd", "macd_sig"]
    return  TradingEnv(
            df=df,
            feature_cols=feature_cols,
            window_size=CONFIG["window_size"],
            initial_balance=CONFIG["initial_balance"],
            commission_pct=CONFIG["commission_pct"],  
            commission_fixed=CONFIG.get("commission_fixed", 0.0),
            spread_pct=CONFIG["spread_pct"],
            slippage_coeff=CONFIG.get("slippage_coeff", 0.1), # replaces slippage_pct
            volume_limit=CONFIG.get("volume_limit", 0.1),
            max_leverage=CONFIG["max_leverage"],
            maintenance_margin=CONFIG.get("maintenance_margin", 0.25),
            financing_rate_annual=CONFIG.get("financing_rate_annual", 0.02),
            reward_scaling=CONFIG.get("reward_scaling", 1.0),
            dd_penalty_coeff=CONFIG.get("dd_penalty_coeff", 0.0),
            turnover_penalty_coeff=CONFIG.get("turnover_penalty_coeff", 0.0),
            normalize_observations=CONFIG.get("normalize_observations", True),
            random_start=CONFIG.get("random_start", True),
            episode_length=CONFIG.get("episode_length", None),
        )



# ========================
# MAIN TRAINING LOGIC
# ========================
def train_ppo_model():
    # 1) Load & prepare data
    if CONFIG["csv_path"] and os.path.exists(CONFIG["csv_path"]):
        df = pd.read_csv(CONFIG["csv_path"])
    else:
        from data_fetcher import download_price_data
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

    # 3) Vectorized environments for parallelism
    env = DummyVecEnv([lambda: make_env(train_df, train_mode=True)])
    eval_env = DummyVecEnv([lambda: make_env(test_df, train_mode=False)])

    # 4) PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=CONFIG["tensorboard_log_dir"],
        ent_coef=0.005,              # entropy regularization for exploration
        learning_rate=3e-4,
        gamma=0.99,                  # discount factor for long-term reward
        gae_lambda=0.95,             # GAE smoothing
        batch_size=256
    )

    # 5) Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=CONFIG["model_save_path"],
        log_path=CONFIG["tensorboard_log_dir"],
        eval_freq=5000,
        deterministic=True,
        render=False,
        callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=1_000, verbose=1)
    )

    # 6) Train
    model.learn(total_timesteps=CONFIG["total_timesteps"], callback=eval_callback)

    # 7) Save
    model.save(os.path.join(CONFIG["model_save_path"], "final_model"))

    print("✅ Training complete. Model saved at:", CONFIG["model_save_path"])
