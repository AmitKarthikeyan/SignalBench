import React, { useEffect, useMemo, useState } from "react";
import { backtest, getMetrics, getTickers, predict, trainModel, ModelName } from "./api";
import Charts from "./components/Charts";
import EquityChart from "./components/EquityChart";

type PredResp = {
  dates: string[];
  close: number[];
  prob_up: number[];
};

export default function App() {
  const [tickers, setTickers] = useState<string[]>([]);
  const [ticker, setTicker] = useState("SPY");
  const [model, setModel] = useState<ModelName>("logreg");

  const today = new Date();
  const yyyy = today.getFullYear();
  const mm = String(today.getMonth() + 1).padStart(2, "0");
  const dd = String(today.getDate()).padStart(2, "0");

  const [start, setStart] = useState(`${yyyy-2}-01-01`);
  const [end, setEnd] = useState(`${yyyy}-${mm}-${dd}`);

  const [lookback, setLookback] = useState<number>(60);
  const [threshold, setThreshold] = useState<number>(0.55);

  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState<string>("");
  const [pred, setPred] = useState<PredResp | null>(null);
  const [metrics, setMetrics] = useState<any>(null);
  const [bt, setBt] = useState<any>(null);

  useEffect(() => {
    getTickers()
      .then((t) => { setTickers(t); if (t.length) setTicker(t[0]); })
      .catch((e) => setMsg(String(e)));
  }, []);

  const predSeries = useMemo(() => {
    if (!pred) return [];
    return pred.dates.map((d, i) => ({
      date: d,
      close: pred.close[i],
      prob_up: pred.prob_up[i]
    }));
  }, [pred]);

  const equitySeries = useMemo(() => {
    if (!bt) return [];
    return bt.dates.map((d: string, i: number) => ({
      date: d,
      equity: bt.equity[i],
      buy_hold: bt.buy_hold[i]
    }));
  }, [bt]);

  async function runTrain() {
    setLoading(true); setMsg("");
    try {
      const r = await trainModel(ticker, model, model === "lstm" ? lookback : undefined);
      setMsg(`Trained ${model} for ${ticker}. Test accuracy: ${r.metrics?.accuracy?.toFixed?.(3) ?? r.metrics?.accuracy}`);
      const m = await getMetrics(ticker, model);
      setMetrics(m);
    } catch (e: any) {
      setMsg(e.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  async function runPredict() {
    setLoading(true); setMsg("");
    try {
      const r = await predict(ticker, model, start, end, model === "lstm" ? lookback : undefined);
      setPred(r);
      const m = await getMetrics(ticker, model).catch(() => null);
      setMetrics(m);
    } catch (e: any) {
      setMsg(e.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  async function runBacktest() {
    setLoading(true); setMsg("");
    try {
      const r = await backtest(ticker, model, start, end, threshold);
      setBt(r);
    } catch (e: any) {
      setMsg(e.message ?? String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="container">
      <div className="card">
        <div className="h1">Stock Forecasting Dashboard</div>
        <div className="muted">Compare scikit-learn baselines vs a PyTorch LSTM on next-day direction.</div>

        <div className="controls" style={{ marginTop: 14 }}>
          <div>
            <label>Ticker</label>
            <select value={ticker} onChange={(e) => setTicker(e.target.value)}>
              {tickers.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </div>

          <div>
            <label>Model</label>
            <select value={model} onChange={(e) => setModel(e.target.value as ModelName)}>
              <option value="logreg">logreg (baseline)</option>
              <option value="gboost">gboost</option>
              <option value="lstm">lstm (PyTorch)</option>
            </select>
          </div>

          <div>
            <label>Start</label>
            <input value={start} onChange={(e) => setStart(e.target.value)} placeholder="YYYY-MM-DD" />
          </div>

          <div>
            <label>End</label>
            <input value={end} onChange={(e) => setEnd(e.target.value)} placeholder="YYYY-MM-DD" />
          </div>

          <div>
            <label>Lookback (LSTM)</label>
            <input
              type="number"
              value={lookback}
              onChange={(e) => setLookback(Number(e.target.value))}
              disabled={model !== "lstm"}
              min={10}
              max={250}
            />
          </div>

          <div style={{ display: "flex", gap: 10 }}>
            <button onClick={runTrain} disabled={loading}>Train</button>
            <button onClick={runPredict} disabled={loading}>Predict</button>
          </div>

          <div>
            <label>Backtest threshold</label>
            <input type="number" step="0.01" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} min={0.5} max={0.9} />
          </div>

          <div style={{ gridColumn: "span 2" }}>
            <button onClick={runBacktest} disabled={loading || !pred}>Run Backtest</button>
          </div>

          <div style={{ gridColumn: "span 3" }}>
            <div className="muted">{loading ? "Working..." : msg}</div>
          </div>
        </div>
      </div>

      {metrics?.metrics && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="h1">Test Metrics (saved from last training)</div>
          <div className="kpis" style={{ marginTop: 10 }}>
            <div className="kpi"><div className="v">{Number(metrics.metrics.accuracy).toFixed(3)}</div><div className="k">Accuracy</div></div>
            <div className="kpi"><div className="v">{Number(metrics.metrics.f1).toFixed(3)}</div><div className="k">F1</div></div>
            <div className="kpi"><div className="v">{metrics.metrics.auc === null ? "—" : Number(metrics.metrics.auc).toFixed(3)}</div><div className="k">AUC</div></div>
            <div className="kpi"><div className="v">{model.toUpperCase()}</div><div className="k">Model</div></div>
          </div>
          <div className="footer">Confusion matrix: {JSON.stringify(metrics.metrics.confusion_matrix)}</div>
        </div>
      )}

      <div className="row" style={{ marginTop: 16 }}>
        <div className="card">
          <div className="h1">Price & Prob(up)</div>
          {predSeries.length ? <Charts series={predSeries} /> : <div className="muted">Click Predict to load data.</div>}
        </div>
        <div className="card">
          <div className="h1">Backtest Equity</div>
          {equitySeries.length ? (
            <>
              <EquityChart series={equitySeries} />
              <div className="footer">
                Total return: {(bt.total_return*100).toFixed(1)}% • Buy&Hold: {(bt.buy_hold_return*100).toFixed(1)}% •
                Sharpe: {bt.sharpe.toFixed(2)} • Max DD: {(bt.max_drawdown*100).toFixed(1)}%
              </div>
            </>
          ) : (
            <div className="muted">Run Backtest after Predict (and Train at least once).</div>
          )}
        </div>
      </div>

      <div className="footer">
        Tip: Train first, then Predict for a window, then Backtest. LSTM may take ~10–30s to train depending on machine.
      </div>
    </div>
  );
}
