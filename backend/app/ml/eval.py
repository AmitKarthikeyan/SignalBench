from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def classification_metrics(y_true: np.ndarray, p1: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (p1 >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, p1)) if len(np.unique(y_true)) > 1 else None,
    }
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    out["confusion_matrix"] = cm.tolist()
    return out

def backtest_long_cash(
    dates: list[str],
    close: np.ndarray,
    p_up: np.ndarray,
    threshold: float = 0.55,
    fee_bps: float = 5.0,
) -> dict:
    # Strategy: if p_up >= threshold, go long for next day; else cash.
    # fee applies when position changes (turnover).
    fee = fee_bps / 10000.0

    # Daily returns from close
    rets = np.diff(np.log(close), prepend=np.log(close[0]))
    # Position for day t: decided at end of day t-1, applied to day t.
    pos = (p_up >= threshold).astype(float)
    # shift by 1 to avoid peeking
    pos = np.roll(pos, 1)
    pos[0] = 0.0

    turnover = np.abs(np.diff(pos, prepend=pos[0]))
    strat_rets = pos * rets - turnover * fee

    equity = np.exp(np.cumsum(strat_rets))
    buyhold = np.exp(np.cumsum(rets))

    # drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    max_dd = float(dd.min())

    # rough annualized sharpe (daily)
    mean = strat_rets.mean()
    std = strat_rets.std()
    sharpe = float((mean / (std + 1e-12)) * np.sqrt(252.0))

    return {
        "dates": dates,
        "equity": equity.tolist(),
        "buy_hold": buyhold.tolist(),
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "total_return": float(equity[-1] - 1.0),
        "buy_hold_return": float(buyhold[-1] - 1.0),
        "threshold": float(threshold),
        "fee_bps": float(fee_bps),
    }
