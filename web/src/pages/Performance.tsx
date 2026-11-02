import { useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Info, Timer, Zap, Layers } from "lucide-react";
import { fmt, int, pct, useApp } from "../data";

const DNN = "#6d4fd8", LR = "#f97316";
const axis = { tick: { fontSize: 11, fill: "#71717a" }, stroke: "#e4e4e7" };

export default function Performance() {
  const { reports: R } = useApp();
  const m = R.metrics, op = m.thresholds.dnn_operating;
  const [t, setT] = useState(op);
  const s = R.sweep.find((x) => Math.abs(x.t - t) < 1e-6)!;
  const cost = s.fn * R.cost_fn + s.fp * R.cost_fp;
  const costCurve = useMemo(() => R.sweep.map((x) => ({ t: x.t, cost: x.fn * R.cost_fn + x.fp * R.cost_fp })), [R]);
  const curve = (k: "roc" | "pr") => (["dnn", "lr"] as const).map((n) => R.curves[n][k].map(([x, y]) => ({ x, [n]: y }))).flat();

  const kpis: [string, keyof typeof m.dnn, string][] = [
    ["ROC-AUC", "roc_auc", "lav"], ["PR-AUC", "pr_auc", "mint"], ["F1", "f1", "cream"],
    ["Precision", "precision", "blue"], ["Recall", "recall", "peach"], ["Accuracy", "accuracy", "gray"],
  ];

  return (
    <>
      <div className="alert info" style={{ marginBottom: 22 }}>
        <Info size={18} style={{ flex: "none", marginTop: 1 }} />
        <span>Held-out test set: <b>{int(m.n_test)}</b> transactions, {pct(m.test_fraud_rate)} fraud
          ({m.dnn.confusion_matrix.tp + m.dnn.confusion_matrix.fn} frauds). Accuracy is uninformative at this prevalence; judge on PR-AUC, precision and recall.</span>
      </div>
      <div className="grid g-6">
        {kpis.map(([lbl, k, cls]) => (
          <div key={k} className={`card stat mini ${cls}`}>
            <div className="val">{fmt(m.dnn[k] as number)}</div>
            <div className="lbl">DNN {lbl}</div>
            <div className="cap" style={{ marginTop: 12, fontWeight: 700 }}>LR {fmt(m.lr[k] as number)}</div>
          </div>
        ))}
      </div>

      <div className="grid g-2 section">
        <ChartCard title="ROC curve" x="False positive rate" y="True positive rate">
          <LineChart data={curve("roc")} margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
            <CartesianGrid stroke="#f1f1f4" />
            <XAxis type="number" dataKey="x" domain={[0, 1]} {...axis} label={{ value: "False positive rate", position: "insideBottom", offset: -10, fontSize: 11, fill: "#71717a" }} />
            <YAxis type="number" domain={[0, 1]} {...axis} />
            <Tooltip formatter={(v: number) => v.toFixed(3)} labelFormatter={(v: number) => `FPR ${v.toFixed(3)}`} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#d4d4d8" strokeDasharray="4 4" />
            <Line dataKey="dnn" name={`DNN · ${fmt(m.dnn.roc_auc)}`} stroke={DNN} dot={false} strokeWidth={2.4} connectNulls />
            <Line dataKey="lr" name={`LR · ${fmt(m.lr.roc_auc)}`} stroke={LR} dot={false} strokeWidth={2.4} connectNulls />
            <Legend verticalAlign="top" height={30} iconType="circle" wrapperStyle={{ fontSize: 12, fontWeight: 600 }} />
          </LineChart>
        </ChartCard>
        <ChartCard title="Precision–recall curve" x="Recall" y="Precision">
          <LineChart data={curve("pr")} margin={{ top: 8, right: 12, bottom: 18, left: 0 }}>
            <CartesianGrid stroke="#f1f1f4" />
            <XAxis type="number" dataKey="x" domain={[0, 1]} {...axis} label={{ value: "Recall", position: "insideBottom", offset: -10, fontSize: 11, fill: "#71717a" }} />
            <YAxis type="number" domain={[0, 1]} {...axis} />
            <Tooltip formatter={(v: number) => v.toFixed(3)} labelFormatter={(v: number) => `Recall ${v.toFixed(3)}`} />
            <Line dataKey="dnn" name={`DNN · ${fmt(m.dnn.pr_auc)}`} stroke={DNN} dot={false} strokeWidth={2.4} connectNulls />
            <Line dataKey="lr" name={`LR · ${fmt(m.lr.pr_auc)}`} stroke={LR} dot={false} strokeWidth={2.4} connectNulls />
            <Legend verticalAlign="top" height={30} iconType="circle" wrapperStyle={{ fontSize: 12, fontWeight: 600 }} />
          </LineChart>
        </ChartCard>
      </div>

      <section className="section">
        <div className="sec-head"><h2>Interactive threshold</h2><span className="hint">DNN · saved test-set probabilities</span></div>
        <div className="card white">
          <div style={{ display: "flex", alignItems: "center", gap: 18, flexWrap: "wrap" }}>
            <div style={{ fontWeight: 800, fontSize: 28, minWidth: 80 }}>{t.toFixed(2)}</div>
            <input type="range" min={0.01} max={0.99} step={0.01} value={t} onChange={(e) => setT(Number(e.target.value))} style={{ flex: 1, minWidth: 200 }} aria-label="Decision threshold" />
            <div className="seg" style={{ background: "var(--soft)", borderRadius: 999, padding: 4 }}>
              <button className={t === op ? "on" : ""} onClick={() => setT(op)}>Operating {op.toFixed(2)}</button>
              <button className={t === m.thresholds.dnn_cost_optimal ? "on" : ""} onClick={() => setT(m.thresholds.dnn_cost_optimal)}>Cost-optimal {m.thresholds.dnn_cost_optimal.toFixed(2)}</button>
            </div>
          </div>
          <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 1.5fr)", marginTop: 22 }}>
            <div className="cm">
              <span /><span style={{ textAlign: "center" }}>Pred legit</span><span style={{ textAlign: "center" }}>Pred fraud</span>
              <span>Actual legit</span><div className="cell" style={{ background: "var(--mint)" }}><b>{int(s.tn)}</b>TN</div><div className="cell" style={{ background: "var(--cream)" }}><b>{int(s.fp)}</b>FP</div>
              <span>Actual fraud</span><div className="cell" style={{ background: "#fde7e7" }}><b>{int(s.fn)}</b>FN</div><div className="cell" style={{ background: "var(--lav)" }}><b>{int(s.tp)}</b>TP</div>
            </div>
            <div>
              <div className="grid g-4" style={{ gap: 12 }}>
                {[["Precision", fmt(s.tp / Math.max(s.tp + s.fp, 1))], ["Recall", fmt(s.tp / Math.max(s.tp + s.fn, 1))],
                  ["Flag rate", pct((s.tp + s.fp) / m.n_test)], ["Expected cost", `$${int(cost)}`]].map(([l, v]) => (
                  <div key={l} className="card mini stat"><div className="val" style={{ fontSize: 20 }}>{v}</div><div className="lbl">{l}</div></div>
                ))}
              </div>
              <div style={{ height: 150, marginTop: 14 }}>
                <ResponsiveContainer>
                  <LineChart data={costCurve} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
                    <XAxis dataKey="t" type="number" domain={[0, 1]} {...axis} />
                    <YAxis {...axis} width={52} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
                    <Tooltip formatter={(v: number) => `$${int(v)}`} labelFormatter={(v: number) => `threshold ${v.toFixed(2)}`} />
                    <ReferenceLine x={t} stroke="#111114" strokeDasharray="3 3" />
                    <Line dataKey="cost" name="Expected cost" stroke={DNN} dot={false} strokeWidth={2.2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="cap">Cost = {s.fn} missed frauds × ${R.cost_fn} + {s.fp} false alarms × ${R.cost_fp}. Approve-all would cost ${int(R.cost.approve_all_cost)}.</div>
            </div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="sec-head"><h2>Latency</h2><span className="hint">Measured on the Python service</span></div>
        <div className="grid g-3">
          <div className="card stat lav"><div className="val">{R.latency.single_no_explanation_ms.p50.toFixed(1)} <small style={{ fontSize: 14 }}>/ {R.latency.single_no_explanation_ms.p95.toFixed(1)} ms</small></div>
            <div className="lbl">Single row, no SHAP (p50 / p95)</div><div className="foot"><span className="tile"><Zap size={19} /></span></div></div>
          <div className="card stat cream"><div className="val">{R.latency.single_with_shap_ms.p50.toFixed(0)} <small style={{ fontSize: 14 }}>/ {R.latency.single_with_shap_ms.p95.toFixed(0)} ms</small></div>
            <div className="lbl">Single row, with SHAP (p50 / p95)</div><div className="foot"><span className="tile"><Timer size={19} /></span></div></div>
          <div className="card stat mint"><div className="val">{int(R.latency.batch.rows_per_second)} <small style={{ fontSize: 14 }}>rows/s</small></div>
            <div className="lbl">Batch throughput</div><div className="foot"><span className="tile"><Layers size={19} /></span></div></div>
        </div>
      </section>
    </>
  );
}

function ChartCard({ title, children }: { title: string; x: string; y: string; children: React.ReactElement }) {
  return (
    <div className="card white">
      <h3 style={{ marginBottom: 8 }}>{title}</h3>
      <div style={{ height: 300 }}><ResponsiveContainer>{children}</ResponsiveContainer></div>
    </div>
  );
}
