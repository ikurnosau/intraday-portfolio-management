import os
from alpaca.trading.client import TradingClient

from core_inference.brokerage_proxies.base_brokerage_proxy import BaseBrokerageProxy


class AlpacaBrokerageProxy(BaseBrokerageProxy):
    def __init__(self): 
        self.trading_client = TradingClient(os.getenv('API_KEY'), os.getenv('API_SECRET'), paper=False)