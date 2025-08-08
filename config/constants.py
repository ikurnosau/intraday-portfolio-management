import pandas as pd


class Constants:
    class Configuration: 
        EXPERIMENT_CONFIGS_DIR = 'experimets/'

    class Data:
        MOST_LIQUID_TECH_STOCKS = [
            "AAPL", "MSFT", "NVDA", "GOOGL", "GOOG",
            "META", "AVGO", "AMD", "TSM", "QCOM",
            "ORCL", "INTC", "CSCO", "IBM", "MU",
            "ADBE", "TXN", "CRM", "PANW", "AMAT",
            "SQ", "PYPL", "NOW", "LRCX", "INTU",
            "ADI", "MCHP", "MRVL", "ON", "KLAC",
            "CDNS", "SNPS", "NXPI", "SMCI", "WDC",
            "STX", "SHOP", "CRWD", "SNOW", "ZS",
            "DDOG", "FTNT", "ANET", "TEAM", "MDB",
            "NET", "OKTA", "PLTR", "HPQ", "DELL"
        ]

        LOWEST_VOL_TO_SPREAD_MAY_JUNE = [
            'AAPL', 'AMD', 'BABA', 'BITU', 'C', 'CSCO', 
            'DAL', 'DIA', 'GLD', 'GOOG', 'IJR', 'MARA', 
            'MRVL', 'MU', 'NEE', 'NKE', 'NVDA', 'ON', 
            'PLTR', 'PYPL', 'QLD', 'QQQ', 'QQQM', 
            'RKLB', 'RSP', 'SMCI', 'SMH', 'SOXL', 
            'SOXX', 'SPXL', 'SPY', 'TMF', 'TNA', 
            'TQQQ', 'TSLA', 'UBER', 'UDOW', 'UPRO', 
            'VOO', 'WFC', 'XBI', 'XLC', 'XLE', 'XLI', 
            'XLK', 'XLU', 'XLV', 'XLY', 'XOM', 'XRT']

        DJIA = [
            # "MMM",  # 3M
            "AXP",  # American Express
            # "AMGN", # Amgen
            # "AMZN", # Amazon
            # "AAPL", # Apple
            # "BA",   # Boeing
            # "CAT",  # Caterpillar
            # "CVX",  # Chevron
            # "CSCO", # Cisco
            # "KO",   # Coca-Cola
            # "DIS",  # Disney
            # "GS",   # Goldman Sachs
            # "HD",   # Home Depot
            # "HON",  # Honeywell
            # "IBM",  # IBM
            # "JNJ",  # Johnson & Johnson
            # "JPM",  # JPMorgan Chase
            # "MCD",  # McDonald's
            # "MRK",  # Merck
            # "MSFT", # Microsoft
            # "NKE",  # Nike
            # "NVDA", # NVIDIA
            # "PG",   # Procter & Gamble
            # "CRM",  # Salesforce
            # "SHW",  # Sherwin-Williams
            # "TRV",  # Travelers
            # "UNH",  # UnitedHealth Group
            # "VZ",   # Verizon
            # "V",    # Visa
            # "WMT"   # Walmart
        ]

        REGULAR_TRADING_HOURS_START = pd.to_datetime("13:30:00").time()
        REGULAR_TRADING_HOURS_END = pd.to_datetime("20:00:00").time()
        
        class Retrieving: 
            class Alpaca:
                BARS_SAVE_DIR = '../data/raw/alpaca/bars'
                BARS_WITH_QUOTES_SAVE_DIR = '../data/raw/alpaca/bars_with_quotes'

    class MLFlow: 
        TRACKING_URI = "http://127.0.0.1:8080"