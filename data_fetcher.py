import yfinance as yf
import pandas as pd
# data_fetcher.py
def download_price_data(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Downloads OHLCV adjusted data via yfinance.
    Example symbol: 'AAPL' or 'EURUSD=X' for forex.
    Returns a DataFrame indexed by datetime with columns: Open, High, Low, Close, Volume, Adj Close if auto_adjust=False.
    """
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data downloaded for {symbol} between {start} and {end}")
    df = df.rename(columns={"Adj Close": "Adj_Close"}) if "Adj Close" in df.columns else df
    df.dropna(inplace=True)
    return df