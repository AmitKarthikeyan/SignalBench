from __future__ import annotations

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime
import time
import hashlib

import numpy as np
import pandas as pd

from .config import settings
from .schemas import (
    TrainRequest, PredictRequest, PredictResponse,
    ExperimentResponse, RunResponse, MetricResponse
)
from . import store
from .core.db import get_db
from .models.experiment import Experiment, Dataset, ModelConfig, Run, Metric, RunStatus
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

def _compute_hash(data: np.ndarray) -> str:
    return hashlib.sha256(data.tobytes()).hexdigest()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/tickers")
def tickers():
    return {"tickers": settings.tickers}

@app.post("/train")
async def train(req: TrainRequest, db: AsyncSession = Depends(get_db)):
    ticker = req.ticker.upper()
    if ticker not in settings.tickers:
        raise HTTPException(400, f"Ticker {ticker} not in supported list.")
    model_name = req.model
    lookback = req.lookback or settings.default_lookback

    # Create or get experiment
    experiment = Experiment(
        name=req.experiment_name or f"{ticker}_{model_name}",
        notes=req.experiment_notes
    )
    db.add(experiment)
    await db.flush()

    # Load data
    feat = _load_features(ticker)
    train_df, test_df = time_split(feat, test_size=req.test_size)
    
    # Create dataset record
    dataset = Dataset(
        provider="yfinance",
        symbol=ticker,
        start_date=settings.data_start,
        end_date=str(feat.index[-1].date()),
        frequency="1d",
        features_version="v1",
        data_hash=_compute_hash(feat[["close"]].to_numpy()),
    )
    db.add(dataset)
    await db.flush()

    # Train model
    train_start = time.time()
    
    if model_name in ("logreg", "gboost"):
        X_train, y_train, _, feature_cols = get_tabular_xy(train_df)
        X_test, y_test, _, _ = get_tabular_xy(test_df)
        
        hyperparams = {}
        training_settings = {"test_size": req.test_size}
        
        model_config = ModelConfig(
            model_type=model_name,
            hyperparameters=hyperparams,
            training_settings=training_settings
        )
        db.add(model_config)
        await db.flush()
        
        run = Run(
            experiment_id=experiment.id,
            dataset_id=dataset.id,
            model_config_id=model_config.id,
            status=RunStatus.RUNNING,
            train_started_at=datetime.utcnow(),
            train_rows=len(train_df),
            test_rows=len(test_df)
        )
        db.add(run)
        await db.flush()
        
        try:
            if model_name == "logreg":
                bundle = train_logreg(X_train, y_train, feature_cols)
            else:
                bundle = train_gboost(X_train, y_train, feature_cols)
            
            store.save_sklearn(ticker, model_name, bundle)
            
            train_time = time.time() - train_start
            
            # Compute metrics
            inference_start = time.time()
            p = predict_proba(bundle, X_test)
            inference_time = (time.time() - inference_start) * 1000 / len(X_test)
            
            m = classification_metrics(y_test, p, threshold=0.5)
            
            run.status = RunStatus.SUCCEEDED
            run.train_finished_at = datetime.utcnow()
            
            metric = Metric(
                run_id=run.id,
                accuracy=m.get("accuracy"),
                f1_score=m.get("f1"),
                roc_auc=m.get("auc"),
                confusion_matrix=m.get("confusion_matrix"),
                train_time_seconds=train_time,
                inference_time_ms=inference_time
            )
            db.add(metric)
            
        except Exception as e:
            run.status = RunStatus.FAILED
            run.error_message = str(e)
            run.train_finished_at = datetime.utcnow()
            await db.commit()
            raise HTTPException(500, f"Training failed: {str(e)}")
    
    else:  # lstm
        X_seq, y_seq, dates, seq_cols = make_sequences(feat, lookback=lookback)
        
        n = len(X_seq)
        split = int(n * (1 - req.test_size))
        X_train, y_train = X_seq[:split], y_seq[:split]
        X_test, y_test = X_seq[split:], y_seq[split:]
        
        val_split = int(len(X_train) * 0.85)
        X_tr, y_tr = X_train[:val_split], y_train[:val_split]
        X_val, y_val = X_train[val_split:], y_train[val_split:]
        
        hyperparams = {"hidden": 64, "layers": 1, "epochs": 8, "lr": 1e-3}
        training_settings = {"lookback": lookback, "test_size": req.test_size}
        
        model_config = ModelConfig(
            model_type=model_name,
            hyperparameters=hyperparams,
            training_settings=training_settings
        )
        db.add(model_config)
        await db.flush()
        
        run = Run(
            experiment_id=experiment.id,
            dataset_id=dataset.id,
            model_config_id=model_config.id,
            status=RunStatus.RUNNING,
            train_started_at=datetime.utcnow(),
            train_rows=split,
            test_rows=n - split
        )
        db.add(run)
        await db.flush()
        
        try:
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
            
            train_time = time.time() - train_start
            
            inference_start = time.time()
            p = predict_lstm(bundle, X_test)
            inference_time = (time.time() - inference_start) * 1000 / len(X_test)
            
            m = classification_metrics(y_test, p, threshold=0.5)
            
            run.status = RunStatus.SUCCEEDED
            run.train_finished_at = datetime.utcnow()
            
            metric = Metric(
                run_id=run.id,
                accuracy=m.get("accuracy"),
                f1_score=m.get("f1"),
                roc_auc=m.get("auc"),
                confusion_matrix=m.get("confusion_matrix"),
                train_time_seconds=train_time,
                inference_time_ms=inference_time
            )
            db.add(metric)
            
        except Exception as e:
            run.status = RunStatus.FAILED
            run.error_message = str(e)
            run.train_finished_at = datetime.utcnow()
            await db.commit()
            raise HTTPException(500, f"Training failed: {str(e)}")
    
    await db.commit()
    
    return {
        "status": "trained",
        "ticker": ticker,
        "model": model_name,
        "experiment_id": experiment.id,
        "run_id": run.id,
        "metrics": {
            "accuracy": metric.accuracy,
            "f1": metric.f1_score,
            "auc": metric.roc_auc
        }
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    ticker = req.ticker.upper()
    model_name = req.model
    lookback = req.lookback or settings.default_lookback

    feat = _load_features(ticker)
    feat = feat.loc[(feat.index >= req.start) & (feat.index <= req.end)].copy()
    if feat.empty:
        raise HTTPException(400, "No rows in requested date range (after feature engineering).")

    dates = feat.index.astype(str).to_list()
    close = feat["close"].astype(float).to_numpy().ravel().tolist()

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
        close=close,
        prob_up=[float(x) for x in p.tolist()],
        pred_up=pred,
    )

@app.get("/experiments", response_model=list[ExperimentResponse])
async def list_experiments(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Experiment, func.count(Run.id).label("run_count"))
        .outerjoin(Run)
        .group_by(Experiment.id)
        .order_by(Experiment.created_at.desc())
    )
    experiments = result.all()
    
    return [
        ExperimentResponse(
            id=exp.id,
            name=exp.name,
            notes=exp.notes,
            created_at=exp.created_at,
            run_count=count
        )
        for exp, count in experiments
    ]

@app.get("/experiments/{experiment_id}/runs", response_model=list[RunResponse])
async def list_runs(experiment_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Run, ModelConfig, Dataset)
        .join(ModelConfig)
        .join(Dataset)
        .where(Run.experiment_id == experiment_id)
        .order_by(Run.created_at.desc())
    )
    runs = result.all()
    
    return [
        RunResponse(
            id=run.id,
            experiment_id=run.experiment_id,
            status=run.status.value,
            model_type=config.model_type,
            symbol=dataset.symbol,
            train_started_at=run.train_started_at,
            train_finished_at=run.train_finished_at,
            created_at=run.created_at
        )
        for run, config, dataset in runs
    ]

@app.get("/runs/{run_id}/metrics", response_model=MetricResponse)
async def get_run_metrics(run_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Metric).where(Metric.run_id == run_id)
    )
    metric = result.scalar_one_or_none()
    
    if not metric:
        raise HTTPException(404, "Metrics not found for this run")
    
    return MetricResponse(
        run_id=metric.run_id,
        accuracy=metric.accuracy,
        f1_score=metric.f1_score,
        roc_auc=metric.roc_auc,
        train_time_seconds=metric.train_time_seconds
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

    pred = predict(PredictRequest(ticker=ticker, model=model, start=start, end=end))
    dates = pred.dates
    close = np.array(pred.close, dtype=np.float32)
    p = np.array(pred.prob_up, dtype=np.float32)

    bt = backtest_long_cash(dates=dates, close=close, p_up=p, threshold=threshold, fee_bps=fee_bps)
    return bt
