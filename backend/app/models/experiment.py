from __future__ import annotations
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Float, Integer, JSON, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from .base import Base, TimestampMixin

class RunStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

class Experiment(Base, TimestampMixin):
    __tablename__ = "experiments"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String(255))
    notes: Mapped[Optional[str]] = mapped_column(Text)
    git_commit: Mapped[Optional[str]] = mapped_column(String(40))
    
    runs: Mapped[list["Run"]] = relationship(back_populates="experiment", cascade="all, delete-orphan")

class Dataset(Base, TimestampMixin):
    __tablename__ = "datasets"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)  # yfinance, stooq, local
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    start_date: Mapped[str] = mapped_column(String(10), nullable=False)
    end_date: Mapped[str] = mapped_column(String(10), nullable=False)
    frequency: Mapped[str] = mapped_column(String(10), nullable=False, default="1d")
    features_version: Mapped[str] = mapped_column(String(50), nullable=False, default="v1")
    data_hash: Mapped[Optional[str]] = mapped_column(String(64))
    feature_hash: Mapped[Optional[str]] = mapped_column(String(64))
    
    runs: Mapped[list["Run"]] = relationship(back_populates="dataset")

class ModelConfig(Base, TimestampMixin):
    __tablename__ = "model_configs"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # logreg, gboost, lstm
    hyperparameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    training_settings: Mapped[dict] = mapped_column(JSON, nullable=False)
    
    runs: Mapped[list["Run"]] = relationship(back_populates="model_config")

class Run(Base, TimestampMixin):
    __tablename__ = "runs"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), nullable=False, index=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    model_config_id: Mapped[int] = mapped_column(ForeignKey("model_configs.id"), nullable=False, index=True)
    
    status: Mapped[RunStatus] = mapped_column(SQLEnum(RunStatus), nullable=False, default=RunStatus.QUEUED)
    train_started_at: Mapped[Optional[datetime]] = mapped_column()
    train_finished_at: Mapped[Optional[datetime]] = mapped_column()
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    train_rows: Mapped[Optional[int]] = mapped_column(Integer)
    test_rows: Mapped[Optional[int]] = mapped_column(Integer)
    
    experiment: Mapped["Experiment"] = relationship(back_populates="runs")
    dataset: Mapped["Dataset"] = relationship(back_populates="runs")
    model_config: Mapped["ModelConfig"] = relationship(back_populates="runs")
    metrics: Mapped[list["Metric"]] = relationship(back_populates="run", cascade="all, delete-orphan")

class Metric(Base, TimestampMixin):
    __tablename__ = "metrics"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"), nullable=False, index=True)
    
    # Classification metrics
    accuracy: Mapped[Optional[float]] = mapped_column(Float)
    f1_score: Mapped[Optional[float]] = mapped_column(Float)
    roc_auc: Mapped[Optional[float]] = mapped_column(Float)
    pr_auc: Mapped[Optional[float]] = mapped_column(Float)
    
    # Calibration metrics
    brier_score: Mapped[Optional[float]] = mapped_column(Float)
    ece: Mapped[Optional[float]] = mapped_column(Float)
    
    # Confusion matrix
    confusion_matrix: Mapped[Optional[dict]] = mapped_column(JSON)
    
    # Timing
    train_time_seconds: Mapped[Optional[float]] = mapped_column(Float)
    inference_time_ms: Mapped[Optional[float]] = mapped_column(Float)
    
    # Additional metrics as JSON
    extra_metrics: Mapped[Optional[dict]] = mapped_column(JSON)
    
    run: Mapped["Run"] = relationship(back_populates="metrics")
