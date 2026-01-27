import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta

def fetch_stock_data(symbol: str, start_date: str, end_date: str, source: str = "yahoo") -> pd.DataFrame:
    """
    Fetch historical stock data from specified source.
    
    Args:
        symbol: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        source: Data source ('yahoo' or 'stooq')
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        if source == "stooq":
            # Use pandas-datareader for Stooq
            # Note: Stooq may require US suffix for US stocks
            stooq_symbol = symbol if '.US' in symbol.upper() else f"{symbol}.US"
            df = web.DataReader(stooq_symbol, "stooq", start=start_date, end=end_date)
            # Stooq returns data in reverse chronological order, so sort it
            df = df.sort_index()
        else:
            # Use yfinance for Yahoo Finance (default)
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
        
        # Check if data was returned
        if df is None or df.empty:
            raise ValueError(f"No data returned for {symbol}.")
        
        # Ensure consistent column naming (lowercase)
        df.columns = df.columns.str.lower()
        
        return df
    except Exception as e:
        # If stooq fails, try falling back to yfinance
        if source == "stooq":
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                if df is None or df.empty:
                    raise ValueError(f"No data returned for {symbol}.")
                df.columns = df.columns.str.lower()
                return df
            except:
                pass
        raise ValueError(f"Failed to fetch data for {symbol} from {source}: {str(e)}")
