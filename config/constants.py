import pandas as pd
from zoneinfo import ZoneInfo

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
            "MMM",  # 3M
            "AXP",  # American Express
            "AMGN", # Amgen
            "AMZN", # Amazon
            "AAPL", # Apple
            "BA",   # Boeing
            "CAT",  # Caterpillar
            "CVX",  # Chevron
            "CSCO", # Cisco
            "KO",   # Coca-Cola
            "DIS",  # Disney
            "GS",   # Goldman Sachs
            "HD",   # Home Depot
            "HON",  # Honeywell
            "IBM",  # IBM
            "JNJ",  # Johnson & Johnson
            "JPM",  # JPMorgan Chase
            "MCD",  # McDonald's
            "MRK",  # Merck
            "MSFT", # Microsoft
            "NKE",  # Nike
            "NVDA", # NVIDIA
            "PG",   # Procter & Gamble
            "CRM",  # Salesforce
            "SHW",  # Sherwin-Williams
            "TRV",  # Travelers
            "UNH",  # UnitedHealth Group
            "VZ",   # Verizon
            "V",    # Visa
            "WMT"   # Walmart
        ]

        DJIA_2000_2004 = sorted("MMM, AXP, XOM, GE, T, BA, CAT, MO, C, KO, DIS, AA, HD, EK, IBM, JNJ, JPM, MCD, MRK, MSFT, HON, INTC, PG, UTX, DD, GM, T, IP, HPQ, WMT".split(', '))
        DJIA_2004_2008 = sorted("MMM, AXP, XOM, GE, T, BA, CAT, MO, C, KO, DIS, AA, HD, PFE, IBM, JNJ, JPM, MCD, MRK, MSFT, HON, INTC, PG, UTX, DD, GM, AIG, VZ, HPQ, WMT".split(', '))
        DJIA_2008_2008 = sorted("MMM, AXP, XOM, GE, T, BA, CAT, CVX, C, KO, DIS, AA, HD, PFE, IBM, JNJ, JPM, MCD, MRK, MSFT, BAC, INTC, PG, UTX, DD, GM, AIG, VZ, HPQ, WMT".split(', '))
        DJIA_2008_2009 = sorted("MMM, AXP, XOM, GE, T, BA, CAT, CVX, C, KO, DIS, AA, HD, PFE, IBM, JNJ, JPM, MCD, MRK, MSFT, BAC, INTC, PG, UTX, DD, GM, KFT, VZ, HPQ, WMT".split(', '))
        DJIA_2009_2012 = sorted("MMM, AXP, XOM, GE, T, BA, CAT, CVX, CSCO, KO, DIS, AA, HD, PFE, IBM, JNJ, JPM, MCD, MRK, MSFT, BAC, INTC, PG, UTX, DD, TRV, KFT, VZ, HPQ, WMT".split(', '))
        DJIA_2012_2013 = sorted("MMM, AXP, XOM, GE, T, BA, CAT, CVX, CSCO, KO, DIS, AA, HD, PFE, IBM, JNJ, JPM, MCD, MRK, MSFT, BAC, INTC, PG, UTX, DD, TRV, UNH, VZ, HPQ, WMT".split(', '))
        DJIA_2013_2015 = sorted("MMM, AXP, XOM, GE, T, BA, CAT, CVX, CSCO, KO, DIS, GS, HD, PFE, IBM, JNJ, JPM, MCD, MRK, MSFT, NKE, INTC, PG, UTX, DD, TRV, UNH, VZ, V, WMT".split(', '))
        DJIA_2015_2017 = sorted("MMM, AXP, XOM, GE, AAPL, BA, CAT, CVX, CSCO, KO, DIS, GS, HD, PFE, IBM, JNJ, JPM, MCD, MRK, MSFT, NKE, INTC, PG, UTX, DD, TRV, UNH, VZ, V, WMT".split(', '))
        DJIA_2017_2018 = sorted("MMM, AXP, XOM, GE, AAPL, BA, CAT, CVX, CSCO, KO, DIS, GS, HD, PFE, IBM, JNJ, JPM, MCD, MRK, MSFT, NKE, INTC, PG, UTX, DWDP, TRV, UNH, VZ, V, WMT".split(', '))
        DJIA_2018_2019 = sorted("MMM, AXP, XOM, WBA, AAPL, BA, CAT, CVX, CSCO, KO, DIS, GS, HD, PFE, IBM, JNJ, JPM, MCD, MRK, MSFT, NKE, INTC, PG, UTX, DWDP, TRV, UNH, VZ, V, WMT".split(', '))

        EASTERN_TZ = ZoneInfo("America/New_York")

        REGULAR_TRADING_HOURS_START = pd.to_datetime("9:30:00" ).time()
        REGULAR_TRADING_HOURS_END = pd.to_datetime("16:00:00").time()
        TRADING_DAY_LENGTH_MINUTES = (REGULAR_TRADING_HOURS_END.hour - REGULAR_TRADING_HOURS_START.hour) * 60 \
            + (REGULAR_TRADING_HOURS_END.minute - REGULAR_TRADING_HOURS_START.minute) \
            + 1
        
        class Retrieving: 
            class Alpaca:
                BARS_SAVE_DIR = '../data/raw/alpaca/bars'
                BARS_WITH_QUOTES_SAVE_DIR = '../data/raw/alpaca/bars_with_quotes'

    class MLFlow: 
        TRACKING_URI = "http://127.0.0.1:8080"