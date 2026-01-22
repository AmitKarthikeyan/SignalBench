# Stock Forecasting Dashboard (FastAPI + React)

A full-stack ML demo project that compares classic ML baselines (scikit-learn) vs a simple deep learning model (PyTorch LSTM) for **next-day direction** prediction on stock/ETF daily data.

## What you get
- **Backend (FastAPI)**: data fetch, feature engineering, train models, predictions, metrics, and a simple backtest.
- **Frontend (React + Vite)**: pick ticker/model/date range, view predictions, metrics, and charts.

## Tech
- Backend: FastAPI, scikit-learn, PyTorch, yfinance, pandas, numpy, joblib
- Frontend: React, TypeScript, Vite, Recharts

---

## Quickstart

### 1) Backend
```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend runs at: `http://localhost:8000`

### 2) Frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: `http://localhost:5173`

---

## API Overview
- `GET /tickers` -> list supported tickers
- `POST /train` -> train a model for a ticker
- `POST /predict` -> predictions for a date range
- `GET /metrics` -> classification metrics on the test split
- `GET /backtest` -> simple long/cash backtest based on probability threshold

---

## Models
- `logreg`: Logistic Regression on engineered features
- `gboost`: Gradient Boosting (HistGradientBoostingClassifier) on engineered features
- `lstm`: PyTorch LSTM on sequences (lookback window)

---

## Project structure
```
stock-ml-dashboard/
  backend/
  frontend/
```

---

## Notes / Tips
- This project uses **time-based splits** to reduce leakage.
- By default, training uses data from 2014-present (based on availability).
- You can add more tickers in `backend/app/config.py`.

