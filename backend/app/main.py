from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd

from .config import settings
from .schemas import TrainRequest, PredictRequest, PredictResponse
from . import store
from .ml.data import fetch_ohlcv, make_features, time_split, get_tabular_xy, make_sequences
from .ml.sk_models import train_logreg, train_gboost, SkModelBundle, predict_proba
from .ml.torch_models import train_lstm, predict_lstm, TorchBundle
from .ml.eval import classification_metrics, backtest_long_cash

app = FastAPI(title="Stock ML Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _load_features(ticker: str) -> pd.DataFrame:
    raw = fetch_ohlcv(ticker, start=settings.data_start)
    feat = make_features(raw)
    return feat

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/tickers")
def tickers():
    return {"tickers": settings.tickers}

@app.post("/train")
def train(req: TrainRequest):
    ticker = req.ticker.upper()
    if ticker not in settings.tickers:
        raise HTTPException(400, f"Ticker {ticker} not in supported list.")
    model_name = req.model
    lookback = req.lookback or settings.default_lookback

    feat = _load_features(ticker)
    train_df, test_df = time_split(feat, test_size=req.test_size)

    if model_name in ("logreg", "gboost"):
        X_train, y_train, _, feature_cols = get_tabular_xy(train_df)
        X_test, y_test, _, _ = get_tabular_xy(test_df)

        if model_name == "logreg":
            bundle = train_logreg(X_train, y_train, feature_cols)
        else:
            bundle = train_gboost(X_train, y_train, feature_cols)

        store.save_sklearn(ticker, model_name, bundle)
        # compute metrics on test for quick confirmation
        p = predict_proba(bundle, X_test)
        m = classification_metrics(y_test, p, threshold=0.5)
        store.save_meta(ticker, model_name, {
            "ticker": ticker,
            "model": model_name,
            "feature_cols": feature_cols,
            "lookback": None,
            "test_size": req.test_size,
            "trained_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "metrics": m,
        })
        return {"status": "trained", "ticker": ticker, "model": model_name, "metrics": m}

    # lstm
    X_seq, y_seq, dates, seq_cols = make_sequences(feat, lookback=lookback)
    # align close series to seq dates
    close = feat.loc[pd.to_datetime(dates)]["close"].to_numpy(dtype=np.float32)

    n = len(X_seq)
    split = int(n * (1 - req.test_size))
    X_train, y_train = X_seq[:split], y_seq[:split]
    X_test, y_test = X_seq[split:], y_seq[split:]

    # simple val split from train tail
    val_split = int(len(X_train) * 0.85)
    X_tr, y_tr = X_train[:val_split], y_train[:val_split]
    X_val, y_val = X_train[val_split:], y_train[val_split:]

    bundle = train_lstm(
        X_tr, y_tr, X_val, y_val,
        n_features=X_seq.shape[-1],
        lookback=lookback,
        feature_cols=seq_cols,
    )
    store.save_torch(ticker, model_name, {
        "state_dict": bundle.state_dict,
        "n_features": bundle.n_features,
        "lookback": bundle.lookback,
        "feature_cols": bundle.feature_cols,
        "test_size": req.test_size,
    })

    p = predict_lstm(bundle, X_test)
    m = classification_metrics(y_test, p, threshold=0.5)
    store.save_meta(ticker, model_name, {
        "ticker": ticker,
        "model": model_name,
        "feature_cols": seq_cols,
        "lookback": lookback,
        "test_size": req.test_size,
        "trained_rows": int(split),
        "test_rows": int(n - split),
        "metrics": m,
    })
    return {"status": "trained", "ticker": ticker, "model": model_name, "metrics": m}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    ticker = req.ticker.upper()
    model_name = req.model
    lookback = req.lookback or settings.default_lookback

    feat = _load_features(ticker)
    # filter by date range
    feat = feat.loc[(feat.index >= req.start) & (feat.index <= req.end)].copy()
    if feat.empty:
        raise HTTPException(400, "No rows in requested date range (after feature engineering).")

    dates = feat.index.astype(str).to_list()
    close = feat["close"].to_numpy(dtype=np.float32)

    if model_name in ("logreg", "gboost"):
        bundle = store.load_sklearn(ticker, model_name)
        if bundle is None:
            raise HTTPException(400, "Model not trained yet. Call /train first.")
        X, _, _, _ = get_tabular_xy(feat)
        p = predict_proba(bundle, X)
    else:
        tstate = store.load_torch(ticker, model_name)
        if tstate is None:
            raise HTTPException(400, "Model not trained yet. Call /train first.")
        lb = int(tstate.get("lookback", lookback))
        X_seq, y_seq, seq_dates, _ = make_sequences(feat, lookback=lb)
        # for sequences, we return aligned dates/close starting at lookback
        dates = seq_dates
        close = feat.loc[pd.to_datetime(dates)]["close"].to_numpy(dtype=np.float32)
        bundle = TorchBundle(
            state_dict=tstate["state_dict"],
            n_features=int(tstate["n_features"]),
            lookback=lb,
            feature_cols=list(tstate["feature_cols"]),
        )
        p = predict_lstm(bundle, X_seq)

    pred = (p >= 0.5).astype(int).tolist()
    return PredictResponse(
        ticker=ticker,
        model=model_name,
        dates=dates,
        close=close.tolist(),
        prob_up=[float(x) for x in p.tolist()],
        pred_up=pred,
    )

@app.get("/metrics")
def metrics(ticker: str, model: str):
    ticker = ticker.upper()
    meta = store.load_meta(ticker, model)
    if meta is None:
        raise HTTPException(404, "No saved metrics. Train first.")
    return meta

@app.get("/backtest")
def backtest(ticker: str, model: str, start: str, end: str, threshold: float = None, fee_bps: float = 5.0):
    ticker = ticker.upper()
    threshold = float(threshold) if threshold is not None else settings.default_threshold

    # reuse /predict to get probabilities for the window
    pred = predict(PredictRequest(ticker=ticker, model=model, start=start, end=end))
    dates = pred.dates
    close = np.array(pred.close, dtype=np.float32)
    p = np.array(pred.prob_up, dtype=np.float32)

    bt = backtest_long_cash(dates=dates, close=close, p_up=p, threshold=threshold, fee_bps=fee_bps)
    return bt
