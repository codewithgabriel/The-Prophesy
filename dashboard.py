# ================================================
# 📂 trading_app/dashboard.py
# ================================================
import plotly.graph_objects as go
import pandas as pd

def plot_equity_curve(trades, initial_balance):
    df = pd.DataFrame(trades)
    df["net_worth"] = initial_balance + (df["shares"] * df["price"]).cumsum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=df["net_worth"], mode="lines", name="Equity Curve"))
    return fig

def plot_trades(df, trades):
    fig = go.Figure(data=[go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])])
    for t in trades:
        fig.add_trace(go.Scatter(x=[df.iloc[t["step"]]["Date"]], y=[t["price"]], mode="markers", marker=dict(color="green" if t["shares"] > 0 else "red"), name="Trade"))
    return fig
