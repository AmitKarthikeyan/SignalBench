from pydantic import BaseModel

class Settings(BaseModel):
    # Keep this list small for demo reliability; you can add more.
    tickers: list[str] = [
        "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META"
    ]
    default_lookback: int = 60
    default_threshold: float = 0.55
    data_start: str = "2014-01-01"

settings = Settings()
