import pandas as pd


REGULAR_TRADING_HOURS_START = pd.to_datetime("14:30:00").time()
REGULAR_TRADING_HOURS_END = pd.to_datetime("21:00:00").time()


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