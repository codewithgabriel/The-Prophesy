import yfinance as yf
import pandas as pd

def download_price_data(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """
    Downloads OHLCV adjusted data via yfinance.
    Example symbol: 'AAPL' or 'EURUSD=X' for forex.
    Returns a DataFrame indexed by datetime with columns: Date, Open, High, Low, Close, Volume, Adj_Close (if applicable).
    """
    try:
        df = yf.download(symbol, start=start, end=end, interval=interval, progress=True, auto_adjust=True)

        if df.empty:
            raise ValueError(f"No data downloaded for {symbol} between {start} and {end}")

        # Flatten MultiIndex columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if col[0] != 'Date' else 'Date' for col in df.columns]

        # Rename Adj Close if present
        df = df.rename(columns={"Adj Close": "Adj_Close"}) if "Adj Close" in df.columns else df

        df.dropna(inplace=True)
        df.columns.name = None  # Remove column name
        df.reset_index(inplace=True)  # Make 'Date' a column
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Failed to download data for {symbol}: {e}")

