
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

    def get_current_balance(self):
        account = self.get_account()
        return float(account['cash'])
    def get_current_price(self, symbol):
        barset = self.api.get_barset(symbol, 'minute', limit=1)
        return barset[symbol][0].c if barset[symbol] else None
    
    def get_historical_data(self, symbol, start, end):
        barset = self.api.get_barset(symbol, 'day', start=start, end=end)
        return barset[symbol] if symbol in barset else []
    def get_positions(self):
        positions = self.api.list_positions()
        return {pos.symbol: pos for pos in positions}
    def close_position(self, symbol):
        position = self.api.get_position(symbol)
        if position:
            return self.api.close_position(symbol)
        return None
    def get_order(self, order_id):
        try:
            return self.api.get_order(order_id)
        except tradeapi.rest.APIError as e:
            print(f"Error fetching order {order_id}: {e}")
            return None
    def cancel_order(self, order_id):
        try:
            return self.api.cancel_order(order_id)
        except tradeapi.rest.APIError as e:
            print(f"Error cancelling order {order_id}: {e}")
            return None
    def get_order_status(self, order_id):
        order = self.get_order(order_id)
        if order:
            return order.status
        return None
    
    def get_trade_history(self, symbol):
        trades = self.api.list_trades(symbol)
        return trades if trades else []

    