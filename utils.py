# ================================================
# 📂 trading_app/utils.py
# ================================================
import pandas as pd

def load_data(csv_path, start_date=None, end_date=None):
    df = pd.read_csv(csv_path)
    print(df.columns.tolist())
    # if start_date:
    #     df = df[df["Date"] >= start_date]
    # if end_date:
    #     df = df[df["Date"] <= end_date]
    # df = df.sort_values("Date").reset_index(drop=True)
    return df

def save_trades_to_csv(trades, filename="trades.csv"):
    pd.DataFrame(trades).to_csv(filename, index=False)


load_data("dataset/aapl.csv", start_date="2020-01-01", end_date="2020-12-31")
