from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

@dataclass
class SkModelBundle:
    model: Any
    feature_cols: list[str]

def train_logreg(X: np.ndarray, y: np.ndarray, feature_cols: list[str]) -> SkModelBundle:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])
    pipe.fit(X, y)
    return SkModelBundle(model=pipe, feature_cols=feature_cols)

def train_gboost(X: np.ndarray, y: np.ndarray, feature_cols: list[str]) -> SkModelBundle:
    clf = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=400,
        random_state=0,
    )
    clf.fit(X, y)
    return SkModelBundle(model=clf, feature_cols=feature_cols)

def predict_proba(bundle: SkModelBundle, X: np.ndarray) -> np.ndarray:
    # Return probability of class 1
    if hasattr(bundle.model, "predict_proba"):
        return bundle.model.predict_proba(X)[:, 1]
    # Fallback: decision_function -> sigmoid
    scores = bundle.model.decision_function(X)
    return 1 / (1 + np.exp(-scores))
