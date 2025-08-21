from envs.trading_env import TradingEnv
import pandas as pd
from envs.config import CONFIG 
from stable_baselines3.common.vec_env import DummyVecEnv

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
            dd_penalty_coeff=CONFIG.get("dd_penalty_coeff", 0.5),
            turnover_penalty_coeff=CONFIG.get("turnover_penalty_coeff", 0.5),
            normalize_observations=CONFIG.get("normalize_observations", True),
            random_start=CONFIG.get("random_start", True),
            episode_length=CONFIG.get("episode_length", None),
        )
