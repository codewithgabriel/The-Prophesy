import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from technical_analysis import add_technical_indicators
from evaluate import sharpe_ratio, max_drawdown, CAGR, compute_returns
from envs.trading_env import TradingEnv

from json import load
with open("../config.json", "r") as f:
    CONFIG = load(f)



def make_eval_env(test_df, window_size=50):
    """
    Wrap environment for evaluation to ensure vectorized shape
    """
    feature_cols = ["Close", "ret", "sma_10", "sma_50", "rsi_14", "macd", "macd_sig"]
    return DummyVecEnv([lambda: TradingEnv(
            df=test_df,
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
    )])

def run_backtest(model_path, test_df, window_size=50):
    """
    Run backtest using trained RL model.
    Returns:
        equity_curve: np.ndarray of net worth values
        trades: trade log (copied during execution)
    """
    env = make_eval_env(test_df, window_size)
    model = PPO.load(model_path)
    
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
        
        # Debug print
        # print(f"Step {len(net_worths)} - Trades: {len(current_trades)}")
    
    # Final verification
    # print(f"\nFinal check - Env trades: {len(getattr(env.envs[0], 'trades', []))}")
    # print(f"Our maintained trades: {len(trades)}")
    
    return np.array(net_worths), trades



def execute_model_backtest():
    if __name__ == "__main__":
        # Load test data
        df = pd.read_csv("/kaggle/input/stock-finance/appl_test.csv")

        # Ensure numeric types for OHLCV
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = add_technical_indicators(df)
        split = int(len(df) * 0.8)
        test_df = df.iloc[split:].reset_index(drop=True)

        # Backtest
        equity, trades = run_backtest("/kaggle/working/ppo_trader_model/best_model.zip", test_df, window_size=50)
        eq_series = pd.Series(equity)
        returns = compute_returns(eq_series)
        
        # === Metrics ===
        print(f"CAGR: {CAGR(eq_series):.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio(returns):.2f}")
        print(f"Max Drawdown: {max_drawdown(eq_series):.2%}")
        print(f"Total Trades: {len(trades)}")
        
        # === Plot equity curve with buy/sell markers ===
        plt.figure(figsize=(12, 6))
        plt.plot(eq_series, label="Equity Curve", color='blue')
        
    
        # if buy_points:
        #     plt.scatter(*zip(*buy_points), marker='^', color='green', label='Buy', s=100)
        # if sell_points:
        #     plt.scatter(*zip(*sell_points), marker='v', color='red', label='Sell', s=100)
        
        plt.title("Equity Curve (Test) with Buy/Sell Markers")
        plt.xlabel("Steps")
        plt.ylabel("Net Worth")
        plt.legend()
        plt.grid(True)
        plt.show()