from __future__ import annotations
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class TrainRequest(BaseModel):
    ticker: str
    model: str = Field(pattern="^(logreg|gboost|lstm)$")
    test_size: float = 0.2
    lookback: int | None = None
    experiment_name: Optional[str] = None
    experiment_notes: Optional[str] = None

class PredictRequest(BaseModel):
    ticker: str
    model: str = Field(pattern="^(logreg|gboost|lstm)$")
    start: str
    end: str
    lookback: int | None = None

class PredictResponse(BaseModel):
    ticker: str
    model: str
    dates: list[str]
    close: list[float]
    prob_up: list[float]
    pred_up: list[int]

class ExperimentResponse(BaseModel):
    id: int
    name: Optional[str]
    notes: Optional[str]
    created_at: datetime
    run_count: int

class RunResponse(BaseModel):
    id: int
    experiment_id: int
    status: str
    model_type: str
    symbol: str
    train_started_at: Optional[datetime]
    train_finished_at: Optional[datetime]
    created_at: datetime

class MetricResponse(BaseModel):
    run_id: int
    accuracy: Optional[float]
    f1_score: Optional[float]
    roc_auc: Optional[float]
    train_time_seconds: Optional[float]
