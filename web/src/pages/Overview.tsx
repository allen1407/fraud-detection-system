import { useMemo, useState } from "react";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ArrowRight, BrainCircuit, ChevronDown, Crosshair, MoreVertical, ShieldAlert, ShieldCheck, ShieldQuestion, SlidersHorizontal, Target, Waves } from "lucide-react";
import { useApp, VERDICT_CLASS, fmt, hhmm, int, money, pct } from "../data";
import type { Verdict } from "../engine";

const VERDICT_ICON = { "AUTO-BLOCK": ShieldAlert, "AUTO-APPROVE": ShieldCheck, "MANUAL REVIEW": ShieldQuestion, "TRANSACTION REJECTED": ShieldQuestion };

export function VerdictChip({ v }: { v: Verdict }) {
  return <span className={`chip ${VERDICT_CLASS[v]}`}><i />{v}</span>;
}

export default function Overview() {
  const { reports: R, samples, go, loadIntoScorer } = useApp();
  const [mode, setMode] = useState<"n" | "fraud">("n");
  const [filter, setFilter] = useState<"all" | Verdict>("all");
  const m = R.metrics;
  const routed = Object.values(m.routing).reduce((a, b) => a + b, 0);
  const rows = samples.recent.filter((r) => filter === "all" || r.verdict === filter);
  const traffic = useMemo(() => R.traffic.map((t) => ({ ...t, label: `Day ${Math.floor(t.h / 24) + 1} · ${String(t.h % 24).padStart(2, "0")}:00` })), [R]);
  const peak = traffic.reduce((a, b) => (b[mode] > a[mode] ? b : a));

  const cards = [
    { cls: "lav", ink: "var(--lav-ink)", val: fmt(m.dnn.roc_auc), lbl: "ROC-AUC · DNN", icon: BrainCircuit, delta: `LR ${fmt(m.lr.roc_auc)}` },
    { cls: "mint", ink: "var(--mint-ink)", val: fmt(m.dnn.pr_auc), lbl: "PR-AUC · DNN", icon: Target, delta: `${(m.dnn.pr_auc / m.test_fraud_rate).toFixed(0)}× base rate` },
    { cls: "cream", ink: "var(--cream-ink)", val: pct(m.dnn.recall, 1), lbl: "Fraud caught · DNN", icon: Crosshair,
      delta: `${m.dnn.confusion_matrix.tp} of ${m.dnn.confusion_matrix.tp + m.dnn.confusion_matrix.fn}` },
  ];

  return (
    <>
      <div className="grid g-main">
        <section>
          <div className="sec-head"><h2>Traffic</h2><span className="hint">Held-out test set · 48 h window</span></div>
          <div className="card blue" style={{ paddingBottom: 12 }}>
            <MoreVertical className="kebab" size={18} />
            <div className="stat">
              <div className="val">{int(m.n_test)} <span style={{ fontSize: 15, fontWeight: 600, color: "var(--ink-2)" }}>transactions</span></div>
              <div className="lbl">{int(R.data.rows)} in the full dataset · {pct(R.data.fraud_rate)} fraud</div>
            </div>
            <div style={{ height: 150, marginTop: 8, marginLeft: -6, marginRight: -6 }}>
              <ResponsiveContainer>
                <AreaChart data={traffic} margin={{ top: 22, right: 6, left: 6, bottom: 0 }}>
                  <defs>
                    <linearGradient id="tg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="h" hide />
                  <YAxis hide domain={[0, "dataMax"]} />
                  <Tooltip cursor={{ stroke: "#93c5fd", strokeDasharray: "3 3" }}
                    content={({ active, payload }) => active && payload?.length ? (
                      <div className="tooltip-dark">{payload[0].payload.label} · {int(payload[0].value as number)} {mode === "n" ? "txns" : "frauds"}</div>
                    ) : null} />
                  <Area type="monotone" dataKey={mode} stroke="#3b82f6" strokeWidth={2.2} fill="url(#tg)"
                    activeDot={{ r: 5, fill: "#3b82f6", stroke: "#fff", strokeWidth: 2 }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
              <div className="seg">
                <button className={mode === "n" ? "on" : ""} onClick={() => setMode("n")}>VOLUME</button>
                <button className={mode === "fraud" ? "on" : ""} onClick={() => setMode("fraud")}>FRAUD</button>
              </div>
              <span className="cap" style={{ fontWeight: 600 }}>Peak {peak.label}</span>
            </div>
          </div>
        </section>

        <section className="stretch">
          <div className="sec-head">
            <h2>Model scorecard</h2>
            <button className="row-btn" onClick={() => go("performance")} aria-label="Open performance"><SlidersHorizontal size={18} /></button>
          </div>
          <div className="grid g-3 g-scroll">
            {cards.map((c) => (
              <div key={c.lbl} className={`card stat ${c.cls}`}>
                <MoreVertical className="kebab" size={18} />
                <div className="val">{c.val}</div>
                <div className="lbl">{c.lbl}</div>
                <div className="foot" style={{ marginTop: 44 }}>
                  <span className="tile"><c.icon size={20} /></span>
                  <span className="delta" style={{ color: c.ink }}>{c.delta}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>

      <div className="grid g-wide section">
        <section>
          <div className="sec-head">
            <h2>Fraud rate is {pct(m.test_fraud_rate)}</h2>
            <div style={{ display: "flex", gap: 8 }}>
              <label className="pill" style={{ position: "relative" }}>
                {filter === "all" ? "All verdicts" : filter} <ChevronDown size={14} />
                <select value={filter} onChange={(e) => setFilter(e.target.value as never)} aria-label="Filter by verdict"
                  style={{ position: "absolute", inset: 0, opacity: 0, cursor: "pointer" }}>
                  <option value="all">All verdicts</option>
                  {Object.keys(m.routing).map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
            </div>
          </div>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead><tr><th>Transaction</th><th className="num">Amount</th><th className="num">DNN risk</th><th className="num">Trust</th><th>Verdict</th><th>Score</th></tr></thead>
              <tbody>
                {rows.map((r) => {
                  const I = VERDICT_ICON[r.verdict];
                  return (
                    <tr key={r.id}>
                      <td><div className="name"><span className="sq"><I size={17} /></span><div>{r.id}<small>{hhmm(Number(r.record.Time))} · {r.label ? "labelled fraud" : "labelled legit"}</small></div></div></td>
                      <td className="num">{money(Number(r.record.Amount))}</td>
                      <td className={`num ${r.dnn_prob >= 0.4 ? "neg" : "pos"}`}>{pct(r.dnn_prob)}</td>
                      <td className="num">{r.trust.toFixed(2)}</td>
                      <td><VerdictChip v={r.verdict} /></td>
                      <td><button className="row-btn" onClick={() => loadIntoScorer(r.record)} aria-label={`Score ${r.id}`}><ArrowRight size={17} /></button></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="grid g-3 g-scroll" style={{ marginTop: 18 }}>
            {Object.entries(m.routing).map(([v, n]) => (
              <div key={v} className="card mini" style={{ background: "#fff", border: "1px solid var(--line)" }}>
                <VerdictChip v={v as Verdict} />
                <div className="stat" style={{ marginTop: 10 }}><div className="val" style={{ fontSize: 22 }}>{int(n)}</div>
                  <div className="lbl">{pct(n / routed, 1)} of test traffic</div></div>
              </div>
            ))}
          </div>
        </section>

        <section>
          <div className="card dark promo">
            <h2>Score a <span className="badge-o">live</span> transaction in seconds</h2>
            <p>A Keras DNN and a logistic regression score it, SHAP explains it, and uncertain cases go to a human analyst. Everything runs in your browser.</p>
            <button className="btn light" onClick={() => go("score")}>Score now <ArrowRight size={16} /></button>
            <svg className="deco" viewBox="0 0 230 200" fill="none" stroke="#fff" strokeWidth="1.4" aria-hidden>
              <rect x="40" y="40" width="170" height="120" rx="16" transform="rotate(-14 125 100)" opacity=".35" />
              <rect x="56" y="58" width="170" height="120" rx="16" transform="rotate(-6 141 118)" opacity=".6" />
              <rect x="72" y="76" width="170" height="120" rx="16" />
              <path d="M100 120h40M100 140h70" opacity=".7" />
              <circle cx="200" cy="112" r="10" opacity=".7" />
            </svg>
          </div>
          <div className="card cream" style={{ marginTop: 22 }}>
            <h3 style={{ display: "flex", alignItems: "center", gap: 8 }}><Waves size={17} /> How decisions are made</h3>
            <p className="cap" style={{ lineHeight: 1.55, margin: "8px 0 0", color: "var(--ink-2)" }}>
              The system acts automatically only when both models agree <b>and</b> the trust score is ≥ 1.0. Everything else is routed
              to <b>manual review</b>. Invalid input is rejected with reasons, never scored.
            </p>
          </div>
        </section>
      </div>
    </>
  );
}
