
# ================================================
# 📂 trading_app/broker_alpaca.py
# ================================================
import alpaca_trade_api as tradeapi

class AlpacaBroker:
    def __init__(self, api_key, api_secret, base_url):
        self.api = tradeapi.REST(api_key, api_secret, base_url)

    def get_account(self):
        return self.api.get_account()._raw

    def place_order(self, symbol, qty, side, type="market", time_in_force="gtc"):
        return self.api.submit_order(symbol=symbol, qty=qty, side=side, type=type, time_in_force=time_in_force)
