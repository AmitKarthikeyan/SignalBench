from __future__ import annotations
from pydantic import BaseModel, Field

class TrainRequest(BaseModel):
    ticker: str
    model: str = Field(pattern="^(logreg|gboost|lstm)$")
    test_size: float = 0.2
    lookback: int | None = None

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
