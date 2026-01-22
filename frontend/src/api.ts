const API_BASE = "http://localhost:8000";

export type ModelName = "logreg" | "gboost" | "lstm";

export async function getTickers(): Promise<string[]> {
  const r = await fetch(`${API_BASE}/tickers`);
  if (!r.ok) throw new Error("Failed to load tickers");
  const j = await r.json();
  return j.tickers;
}

export async function trainModel(ticker: string, model: ModelName, lookback?: number) {
  const r = await fetch(`${API_BASE}/train`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ ticker, model, lookback })
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || "Train failed");
  return j;
}

export async function getMetrics(ticker: string, model: ModelName) {
  const r = await fetch(`${API_BASE}/metrics?ticker=${encodeURIComponent(ticker)}&model=${encodeURIComponent(model)}`);
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || "Metrics failed");
  return j;
}

export async function predict(ticker: string, model: ModelName, start: string, end: string, lookback?: number) {
  const r = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ ticker, model, start, end, lookback })
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || "Predict failed");
  return j;
}

export async function backtest(ticker: string, model: ModelName, start: string, end: string, threshold: number) {
  const r = await fetch(`${API_BASE}/backtest?ticker=${encodeURIComponent(ticker)}&model=${encodeURIComponent(model)}&start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}&threshold=${threshold}`);
  const j = await r.json();
  if (!r.ok) throw new Error(j.detail || "Backtest failed");
  return j;
}
