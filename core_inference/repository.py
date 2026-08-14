import logging
import pandas as pd
import time
from typing import Any

from data.raw.retrievers.alpaca_markets_retriever import AlpacaMarketsRetriever


class Repository:
    def __init__(self, 
                 trading_symbols: list[str], 
                 required_history_depth: int = 500,
                 bars_and_quotes: dict[str: pd.DataFrame] | None = None,
                 retriever: AlpacaMarketsRetriever | None = None):
        self.symbols = sorted(trading_symbols)
        self.retriever = retriever
        self.required_history_depth = required_history_depth
        self.bars_and_quotes = bars_and_quotes if bars_and_quotes is not None \
            else self.initialize_bars_and_quotes_with_latest_values(required_history_depth)

        self.bid_price =  {symbol: self.bars_and_quotes[symbol]['bid_price'].iloc[-1] for symbol in trading_symbols}
        self.ask_price = {symbol: self.bars_and_quotes[symbol]['ask_price'].iloc[-1] for symbol in trading_symbols}
        self.bid_size =  {symbol: self.bars_and_quotes[symbol]['bid_size'].iloc[-1] for symbol in trading_symbols}
        self.ask_size = {symbol: self.bars_and_quotes[symbol]['ask_size'].iloc[-1] for symbol in trading_symbols}
        self.quote_timestamp = {symbol: None for symbol in trading_symbols}
        self.quote_received_at = {symbol: None for symbol in trading_symbols}
        self.bar_received_at = {symbol: None for symbol in trading_symbols}

    def initialize_bars_and_quotes_with_latest_values(self, bars_history_depth: int):
        print('Starting bars and quotes initialization...')

        latest_quotes: dict[str, dict[str: float]] = self.retriever.latest_quote(self.symbols)
        latest_bars: dict[str, pd.DataFrame] = self.retriever.latest_bars(self.symbols, limit=bars_history_depth)
        for symbol, bar_df in latest_bars.items():
            bar_df['bid_price'] = latest_quotes[symbol]['bid_price']
            bar_df['ask_price'] = latest_quotes[symbol]['ask_price']
            bar_df['bid_size'] = latest_quotes[symbol]['bid_size']
            bar_df['ask_size'] = latest_quotes[symbol]['ask_size']

        logging.info('Bars and quotes initialization finished!')

        return latest_bars

    def add_bar(self, data: dict[str: Any]):
        self.bar_received_at[data["symbol"]] = time.time()
        self.bars_and_quotes[data["symbol"]].loc[len(self.bars_and_quotes[data["symbol"]])] = {
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"],
            "date": data["date"], 
            "bid_price": data['bid_price'] if 'bid_price' in data else self.bid_price[data["symbol"]],
            "bid_size": data['bid_size'] if 'bid_size' in data else self.bid_size[data["symbol"]],
            "ask_price": data['ask_price'] if 'ask_price' in data else self.ask_price[data["symbol"]],
            "ask_size": data['ask_size'] if 'ask_size' in data else self.ask_size[data["symbol"]],
        }

    def update_quote(self, data):
        self.bid_price[data.symbol] = data.bid_price
        self.ask_price[data.symbol] = data.ask_price
        self.bid_size[data.symbol] = data.bid_size
        self.ask_size[data.symbol] = data.ask_size
        self.quote_timestamp[data.symbol] = getattr(data, "timestamp", None)
        self.quote_received_at[data.symbol] = time.time()

    def get_asset_dfs(self) -> dict[str: pd.DataFrame]:
        return {symbol: self.bars_and_quotes[symbol].tail(self.required_history_depth) for symbol in self.symbols}

    def get_latest_asset_data(self, symbol: str) -> dict[str, Any]:
        latest_data = self.bars_and_quotes[symbol].iloc[-1].to_dict()
        bid = float(self.bid_price[symbol])
        ask = float(self.ask_price[symbol])
        now = time.time()
        quote_received_at = self.quote_received_at[symbol]
        bar_received_at = self.bar_received_at[symbol]
        latest_data.update({
            "symbol": symbol,
            "bid_price": bid,
            "ask_price": ask,
            "midpoint": (bid + ask) / 2,
            "bid_size": float(self.bid_size[symbol]),
            "ask_size": float(self.ask_size[symbol]),
            "quote_timestamp": self.quote_timestamp[symbol],
            "quote_received_at": quote_received_at,
            "quote_age_ms": (
                (now - quote_received_at) * 1000
                if quote_received_at is not None
                else None
            ),
            "bar_received_at": bar_received_at,
            "bar_age_ms": (
                (now - bar_received_at) * 1000
                if bar_received_at is not None
                else None
            ),
        })
        return latest_data

    def get_bid_price(self, symbol):
        return self.bid_price[symbol]

    def get_ask_price(self, symbol):
        return self.ask_price[symbol]

    def get_symbols(self) -> list[str]:
        return self.symbols