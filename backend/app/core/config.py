from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://signalbench:signalbench@localhost:5432/signalbench"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # yfinance configuration
    YF_USE_CURL_CFFI: str = "1"
    
    # Data settings
    data_start: str = "2020-01-01"
    tickers: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    # ML settings
    default_lookback: int = 20
    default_threshold: float = 0.5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Set environment variable for yfinance before creating settings instance
os.environ.setdefault("YF_USE_CURL_CFFI", "1")

settings = Settings()

# Ensure the environment variable is set
os.environ["YF_USE_CURL_CFFI"] = settings.YF_USE_CURL_CFFI
