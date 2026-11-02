import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Info } from "lucide-react";
import { fmt, int, pct, useApp } from "../data";
import { VerdictChip } from "./Overview";
import type { Verdict } from "../engine";

const axis = { tick: { fontSize: 11, fill: "#71717a" }, stroke: "#e4e4e7" };
const PALETTE = ["#6d4fd8", "#2563eb", "#16a34a", "#d97706", "#e5484d"];

export default function Responsible() {
  const { reports: R } = useApp();
  const fair = [["By amount band", R.fairness_amount], ["By time of day", R.fairness_time]] as const;
  const isGap = (r: Record<string, unknown>) => String(r.group).toUpperCase() === "MAX GAP";
  const nFraud = fair.flatMap(([, d]) => d.filter((r) => !isGap(r)).map((r) => Number(r.n_fraud)));
  const rs = R.robustness_summary, ss = R.shap_stability;
  const shap = R.shap_global.slice(0, 12);

  return (
    <>
      <section>
        <div className="sec-head"><h2>Fairness & subgroup performance</h2><span className="hint">DNN at the operating threshold</span></div>
        <div className="alert info" style={{ marginBottom: 18 }}>
          <Info size={18} style={{ flex: "none", marginTop: 1 }} />
          <span>The dataset has no customer attributes (gender, age, location), so protected-group fairness cannot be audited.
            As a proxy, performance is compared across transaction segments.</span>
        </div>
        <div className="grid g-2">
          {fair.map(([title, d]) => (
            <div key={title} className="card white">
              <div className="sec-head" style={{ marginBottom: 0, minHeight: 0 }}><h3>{title}</h3><span className="hint">Recall by segment</span></div>
              <div style={{ height: 200, marginTop: 10 }}>
                <ResponsiveContainer>
                  <BarChart data={d.filter((r) => !isGap(r))} margin={{ top: 6, right: 6, bottom: 0, left: -18 }}>
                    <CartesianGrid stroke="#f1f1f4" vertical={false} />
                    <XAxis dataKey="group" {...axis} interval={0} />
                    <YAxis domain={[0, 1]} {...axis} />
                    <Tooltip formatter={(v: number) => fmt(v)} cursor={{ fill: "#f6f7fb" }} />
                    <Bar isAnimationActive={false} dataKey="tpr" name="Recall (TPR)" radius={[8, 8, 0, 0]}>
                      {d.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="tbl-wrap" style={{ marginTop: 10 }}>
                <table className="tbl lined">
                  <thead><tr><th>Segment</th><th className="num">n</th><th className="num">Frauds</th><th className="num">Recall</th><th className="num">FPR</th><th className="num">Precision</th></tr></thead>
                  <tbody>{d.map((r) => (
                    <tr key={String(r.group)} style={isGap(r) ? { background: "var(--soft)" } : undefined}><td>{isGap(r) ? "Max gap" : String(r.group)}</td><td className="num">{isGap(r) ? "–" : int(Number(r.n))}</td><td className="num">{isGap(r) ? "–" : String(r.n_fraud)}</td>
                      <td className="num">{fmt(Number(r.tpr))}</td><td className="num">{pct(Number(r.fpr), 3)}</td><td className="num">{r.precision == null ? "–" : fmt(Number(r.precision))}</td></tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
        <div className="cap" style={{ marginTop: 10 }}>Each segment holds only {Math.min(...nFraud)}–{Math.max(...nFraud)} frauds, so a recall gap mostly reflects a handful of transactions.
          Gaps above the 0.20 audit criterion should be re-checked on more data before deployment.</div>
      </section>

      <section className="section">
        <div className="sec-head"><h2>Robustness & explainability</h2></div>
        <div className="grid g-4">
          <div className="card stat lav"><div className="val">{pct(rs.dnn_flip_rate, 0)}</div><div className="lbl">DNN class flipped</div>
            <div className="cap" style={{ marginTop: 14 }}>{rs.n_cases} counterfactual searches</div></div>
          <div className="card stat blue"><div className="val">{pct(rs.verdict_flip_rate, 0)}</div><div className="lbl">Verdict flipped</div>
            <div className="cap" style={{ marginTop: 14 }}>{rs.no_flip_cases}/{rs.n_cases} never flipped within {rs.search_max_std.toFixed(1)} SD</div></div>
          <div className="card stat mint"><div className="val">{fmt(ss.mean_spearman)}</div><div className="lbl">SHAP rank stability</div>
            <div className="cap" style={{ marginTop: 14 }}>± {fmt(ss.std_spearman)} Spearman · 1% noise</div></div>
          <div className="card stat cream"><div className="val">{fmt(ss.min_spearman)}</div><div className="lbl">Worst-case stability</div>
            <div className="cap" style={{ marginTop: 14 }}>min over {ss.n_samples} noisy samples</div></div>
        </div>
        <div className="grid g-wide" style={{ marginTop: 22 }}>
          <div className="card white">
            <h3>Global feature importance <span className="cap">· mean |SHAP|, DNN</span></h3>
            <div style={{ height: 340, marginTop: 10 }}>
              <ResponsiveContainer>
                <BarChart data={shap} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 0 }}>
                  <CartesianGrid stroke="#f1f1f4" horizontal={false} />
                  <XAxis type="number" {...axis} tickFormatter={(v) => v.toExponential(1)} />
                  <YAxis type="category" dataKey="feature" {...axis} width={70} />
                  <Tooltip formatter={(v: number) => v.toExponential(2)} cursor={{ fill: "#f6f7fb" }} />
                  <Bar isAnimationActive={false} dataKey="mean_abs_shap" name="mean |SHAP|" fill="#6d4fd8" radius={[0, 8, 8, 0]} barSize={16} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="card white">
            <h3>Robustness cases</h3>
            <div className="tbl-wrap" style={{ marginTop: 10, maxHeight: 340, overflow: "auto" }}>
              <table className="tbl lined">
                <thead><tr><th>#</th><th>Label</th><th className="num">P(fraud)</th><th>Verdict</th></tr></thead>
                <tbody>{R.robustness.map((r) => (
                  <tr key={String(r.case)}><td className="cap">{String(r.case)}</td><td>{Number(r.is_fraud) ? "Fraud" : "Legit"}</td>
                    <td className="num">{pct(Number(r.original_prob))}</td><td><VerdictChip v={r.original_verdict as Verdict} /></td></tr>
                ))}</tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="sec-head"><h2>Human-in-the-loop</h2><span className="hint">Why manual review exists</span></div>
        <div className="steps">
          {[["lav", "Both models agree", "The DNN and logistic regression both flag, or both clear, the transaction."],
            ["blue", "Trust score ≥ 1.0", "The transaction is as familiar as typical training data (kNN distance)."],
            ["cream", "Otherwise: review", "Disagreement or an unfamiliar transaction goes to a human fraud analyst."],
            ["gray", "Invalid input", "Missing, non-numeric or out-of-range values are rejected with reasons, never scored."]]
            .map(([c, h, p], i) => <div key={h} className={`card ${c}`}><div className="n">{i + 1}</div><h3>{h}</h3><p>{p}</p></div>)}
        </div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 16 }}>
          {Object.entries(R.metrics.routing).map(([k, v]) => (
            <span key={k} className="pill" style={{ cursor: "default" }}><VerdictChip v={k as Verdict} /> {int(v)}</span>
          ))}
        </div>
      </section>
    </>
  );
}
