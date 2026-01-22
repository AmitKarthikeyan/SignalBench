import os
from pathlib import Path
import joblib
import torch

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
SK_DIR = ARTIFACT_DIR / "sklearn"
TORCH_DIR = ARTIFACT_DIR / "torch"
META_DIR = ARTIFACT_DIR / "meta"

for d in (SK_DIR, TORCH_DIR, META_DIR):
    d.mkdir(parents=True, exist_ok=True)

def sk_path(ticker: str, model: str) -> Path:
    return SK_DIR / f"{ticker}__{model}.joblib"

def torch_path(ticker: str, model: str) -> Path:
    return TORCH_DIR / f"{ticker}__{model}.pt"

def meta_path(ticker: str, model: str) -> Path:
    return META_DIR / f"{ticker}__{model}.json"

def save_sklearn(ticker: str, model: str, obj) -> None:
    joblib.dump(obj, sk_path(ticker, model))

def load_sklearn(ticker: str, model: str):
    p = sk_path(ticker, model)
    if not p.exists():
        return None
    return joblib.load(p)

def save_torch(ticker: str, model: str, state: dict) -> None:
    torch.save(state, torch_path(ticker, model))

def load_torch(ticker: str, model: str):
    p = torch_path(ticker, model)
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu")

def save_meta(ticker: str, model: str, meta: dict) -> None:
    import json
    with open(meta_path(ticker, model), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def load_meta(ticker: str, model: str):
    import json
    p = meta_path(ticker, model)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
