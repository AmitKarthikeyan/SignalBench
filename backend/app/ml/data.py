from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

def fetch_ohlcv(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker}.")
    df = df.rename(columns=str.lower)
    # yfinance columns typically: open high low close adj close volume
    # keep required
    keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
    df = df[keep].copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # log returns
    out["ret_1"] = np.log(out["close"]).diff()

    # ranges and volume transforms
    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    out["oc_range"] = (out["close"] - out["open"]) / out["open"]
    out["log_vol"] = np.log(out["volume"].replace(0, np.nan))

    # rolling stats on returns
    for w in (5, 10, 20):
        out[f"ret_mean_{w}"] = out["ret_1"].rolling(w).mean()
        out[f"ret_std_{w}"] = out["ret_1"].rolling(w).std()
        out[f"mom_{w}"] = out["close"].pct_change(w)

    # lag returns
    for k in range(1, 21):
        out[f"ret_lag_{k}"] = out["ret_1"].shift(k)

    # target: next-day direction
    out["target_up"] = (out["ret_1"].shift(-1) > 0).astype(int)

    out = out.dropna().copy()
    return out

def time_split(df_feat: pd.DataFrame, test_size: float = 0.2):
    n = len(df_feat)
    if n < 300:
        raise ValueError("Not enough data after feature engineering (need ~300+ rows).")
    split = int(n * (1 - test_size))
    train = df_feat.iloc[:split].copy()
    test = df_feat.iloc[split:].copy()
    return train, test

def get_tabular_xy(df_feat: pd.DataFrame):
    feature_cols = [c for c in df_feat.columns if c not in ("target_up",)]
    X = df_feat[feature_cols].to_numpy(dtype=np.float32)
    y = df_feat["target_up"].to_numpy(dtype=np.int64)
    dates = df_feat.index.astype(str).to_list()
    return X, y, dates, feature_cols

def make_sequences(df_feat: pd.DataFrame, lookback: int):
    # Use per-day inputs: [ret_1, hl_range, oc_range, log_vol, ret_mean_5, ret_std_5, mom_5]
    cols = ["ret_1", "hl_range", "oc_range", "log_vol", "ret_mean_5", "ret_std_5", "mom_5"]
    for c in cols:
        if c not in df_feat.columns:
            raise ValueError(f"Missing expected column {c}")

    values = df_feat[cols].to_numpy(dtype=np.float32)
    y = df_feat["target_up"].to_numpy(dtype=np.int64)
    dates = df_feat.index.astype(str).to_list()

    X_seq = []
    y_seq = []
    d_seq = []
    for i in range(lookback, len(df_feat)):
        X_seq.append(values[i-lookback:i])
        y_seq.append(y[i])
        d_seq.append(dates[i])
    X_seq = np.stack(X_seq).astype(np.float32)
    y_seq = np.array(y_seq, dtype=np.int64)
    return X_seq, y_seq, d_seq, cols
