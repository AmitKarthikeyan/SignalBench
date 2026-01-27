# backend/app/universe.py
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

WIKI_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.json")

def _load_cache(name: str, max_age_hours: int = 24) -> Optional[List[str]]:
    path = _cache_path(name)
    if not os.path.exists(path):
        return None
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    if datetime.now() - mtime > timedelta(hours=max_age_hours):
        return None
    with open(path, "r") as f:
        return json.load(f)

def _save_cache(name: str, tickers: List[str]) -> None:
    with open(_cache_path(name), "w") as f:
        json.dump(tickers, f)

def _read_html_with_headers(url: str) -> list[pd.DataFrame]:
    """
    Wikipedia sometimes returns 403 to default Python user agents.
    Fetch HTML with a browser-like User-Agent, then let pandas parse the tables.
    """
    req = urllib.request.Request(url, headers={"User-Agent": WIKI_UA})
    with urllib.request.urlopen(req) as resp:
        html = resp.read()
    return pd.read_html(html)

def _clean_tickers(tickers: list[str]) -> list[str]:
    # BRK.B style tickers often appear with ".", convert to "-" for yfinance
    cleaned = (
        pd.Series(tickers, dtype="string")
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(".", "-", regex=False)
        .tolist()
    )
    return sorted({t for t in cleaned if t and t.lower() != "nan"})

def get_sp500_tickers() -> List[str]:
    cached = _load_cache("sp500")
    if cached:
        return cached

    tables = _read_html_with_headers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = tables[0]
    if "Symbol" not in df.columns:
        raise ValueError("Could not parse S&P 500 tickers: missing 'Symbol' column.")
    tickers = _clean_tickers(df["Symbol"].astype(str).tolist())

    _save_cache("sp500", tickers)
    return tickers

def get_nasdaq100_tickers() -> List[str]:
    cached = _load_cache("nasdaq100")
    if cached:
        return cached

    tables = _read_html_with_headers("https://en.wikipedia.org/wiki/Nasdaq-100")

    df = None
    ticker_col = None

    # Find a table with a ticker/symbol column (Wikipedia sometimes changes layout)
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        cols_lower = [c.lower() for c in cols]
        for candidate in ("ticker", "ticker symbol", "symbol", "stock symbol"):
            if candidate in cols_lower:
                df = t
                ticker_col = cols[cols_lower.index(candidate)]
                break
        if df is not None:
            break

    if df is None or ticker_col is None:
        raise ValueError("Could not find Nasdaq-100 tickers table (ticker/symbol column not found).")

    tickers = _clean_tickers(df[ticker_col].astype(str).tolist())

    _save_cache("nasdaq100", tickers)
    return tickers

def get_universe(name: str) -> List[str]:
    name = (name or "default").lower()
    if name in ("default", "starter"):
        return ["SPY", "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "NVDA", "META"]
    if name in ("sp500", "s&p500", "spx"):
        return get_sp500_tickers()
    if name in ("nasdaq100", "ndx", "nas100"):
        return get_nasdaq100_tickers()
    raise ValueError(f"Unknown universe: {name}")
