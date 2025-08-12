import numpy as np
import pandas as pd

def compute_returns(series: pd.Series) -> pd.Series:
    """
    Compute simple percentage returns.
    Args:
        series: Pandas Series of equity or price values.
    Returns:
        Series of percentage returns.
    """
    return series.pct_change().fillna(0)

def sharpe_ratio(returns: pd.Series, freq: int = 252, rf: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.
    Args:
        returns: Series of returns.
        freq: Periods per year (252 for daily, 12 for monthly).
        rf: Risk-free rate (annualized).
    """
    ann_ret = returns.mean() * freq
    ann_vol = returns.std() * np.sqrt(freq)
    return np.nan if ann_vol == 0 else (ann_ret - rf) / ann_vol

def max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum drawdown over the equity curve.
    Args:
        equity_curve: Pandas Series of equity values.
    """
    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max
    return drawdown.min()

def CAGR(equity_series: pd.Series, days_per_year: int = 252) -> float:
    """
    Compound Annual Growth Rate.
    """
    start, end = equity_series.iloc[0], equity_series.iloc[-1]
    n_days = len(equity_series)
    return (end / start) ** (days_per_year / n_days) - 1
