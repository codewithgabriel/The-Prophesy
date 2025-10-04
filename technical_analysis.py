import pandas as pd
import numpy as np

import lib_ta.pandas_ta as ta



def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhances price DataFrame with a rich set of technical indicators.
    Designed for RL trading environments — avoids NaNs in observations
    by careful filling.

    Features added:
        - Returns (daily % change)
        - Simple Moving Averages (SMA)
        - Exponential Moving Averages (EMA)
        - Relative Strength Index (RSI)
        - Moving Average Convergence Divergence (MACD + Signal)
        - Average True Range (ATR) for volatility
        - Bollinger Band width
        - On-Balance Volume (OBV) for volume flow
    """

    df = df.copy()
    print(f"[INFO] Raw dataset shape: {df.shape}")

    # Ensure Close is numeric
    df["Close"] = pd.to_numeric(df["Close"], errors='coerce')
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce').fillna(0)
    else:
        df["Volume"] = 0  # fallback if no volume data

    # 1. Price Returns
    df["ret"] = df["Close"].pct_change().fillna(0)

    # 2. Trend Indicators
    df["sma_10"] = df["Close"].rolling(10, min_periods=1).mean()
    df["sma_50"] = df["Close"].rolling(50, min_periods=1).mean()
    df["ema_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean()

    # 3. Momentum
    df["rsi_14"] = ta.rsi(df["Close"], length=14)

    # 4. MACD
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_sig"] = macd["MACDs_12_26_9"]
    else:
        df["macd"] = 0
        df["macd_sig"] = 0

    # 5. Volatility
    df["atr_14"] = ta.atr(df["High"], df["Low"], df["Close"], length=14) if all(
        col in df.columns for col in ["High", "Low"]
    ) else 0

    # 6. Bollinger Bands (width only to reduce col count)
    bb = ta.bbands(df["Close"], length=20, std=2)
    if bb is not None and not bb.empty:
        df["bb_width"] = (bb["BBU_20_2.0"] - bb["BBL_20_2.0"]) / df["Close"]
    else:
        df["bb_width"] = 0

    # 7. Volume Flow
    df["obv"] = ta.obv(df["Close"], df["Volume"]) if "Volume" in df.columns else 0

    # 8. Fill NaNs conservatively
    df.fillna(method="bfill", inplace=True)
    df.fillna(method="ffill", inplace=True)

    # 9. Final safety: replace any remaining NaNs
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    print(f"[INFO] Feature-enhanced dataset shape: {df.shape}")
    return df
