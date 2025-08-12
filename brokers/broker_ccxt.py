# ================================================
# 📂 trading_app/broker_ccxt.py
# ================================================
import ccxt

class CCXTBroker:
    def __init__(self, exchange_id, api_key, secret):
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({"apiKey": api_key, "secret": secret})

    def get_balance(self):
        return self.exchange.fetch_balance()

    def place_order(self, symbol, side, amount, price=None):
        if price:
            return self.exchange.create_limit_order(symbol, side, amount, price)
        else:
            return self.exchange.create_market_order(symbol, side, amount)