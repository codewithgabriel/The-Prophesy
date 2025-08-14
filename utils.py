# ================================================
# 📂 trading_app/utils.py
# ================================================
import pandas as pd
from envs.config import CONFIG
from technical_analysis import add_technical_indicators
import os
from data_fetcher import download_price_data


def load_data(save=False):
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
    if save:
        df.to_csv(CONFIG["csv_path"], index=False)
        print("Data saved to:", CONFIG["csv_path"])
    return df

def save_trades_to_csv(trades, filename="trades.csv"):
    pd.DataFrame(trades).to_csv(filename, index=False)

