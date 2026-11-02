import { useEffect, useState } from "react";
import { ChevronDown, Download, FileText, Loader2, ScanSearch, ShieldAlert, ShieldCheck, ShieldQuestion, ShieldX, Sparkles } from "lucide-react";
import { download, pct, useApp, VERDICT_COLOR } from "../data";
import type { Rec, Result } from "../engine";

const PCA = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);
const ICON = { "AUTO-BLOCK": ShieldAlert, "AUTO-APPROVE": ShieldCheck, "MANUAL REVIEW": ShieldQuestion, "TRANSACTION REJECTED": ShieldX };
const BG = { "AUTO-BLOCK": "#fde7e7", "AUTO-APPROVE": "var(--mint)", "MANUAL REVIEW": "var(--cream)", "TRANSACTION REJECTED": "var(--gray)" };

const toForm = (r: Rec) => Object.fromEntries(Object.entries(r).map(([k, v]) => [k, String(Number(Number(v).toFixed(6)))]));

export default function Score() {
  const { samples, engine, pending, clearPending } = useApp();
  const names = Object.keys(samples.presets);
  const [preset, setPreset] = useState<string | null>(pending ? null : names[0]);
  const [form, setForm] = useState<Record<string, string>>(() => toForm(pending ?? samples.presets[names[0]]));
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<Result | null>(null);
  const [showReport, setShowReport] = useState(false);

  useEffect(() => { if (pending) clearPending(); }, [pending, clearPending]);

  const choose = (name: string) => {
    setPreset(name); setRes(null);
    setForm(name === "Invalid Input" ? { ...toForm(samples.presets[names[0]]), Amount: "-250" } : toForm(samples.presets[name]));
  };
  const set = (k: string, v: string) => { setForm((f) => ({ ...f, [k]: v })); setPreset(null); };

  const predict = () => {
    if (!engine) return;
    setBusy(true);
    setTimeout(() => { setRes(engine.assess(form)); setBusy(false); setShowReport(false); }, 30);   // let the spinner paint
  };

  return (
    <div className="grid g-main" style={{ alignItems: "start" }}>
      <section className="card" style={{ background: "var(--soft)" }}>
        <div className="sec-head" style={{ marginBottom: 10 }}><h2 style={{ fontSize: 17 }}>Transaction</h2><span className="hint">Real test-set rows</span></div>
        <div className="seg" style={{ background: "var(--gray)", borderRadius: 999, padding: 4, marginBottom: 20 }}>
          {[...names, "Invalid Input"].map((n) => <button key={n} className={preset === n ? "on" : ""} onClick={() => choose(n)}>{n}</button>)}
        </div>
        <div className="grid g-2" style={{ gap: 14 }}>
          <label className="field">Time <span className="h">seconds since window start (0–172,800)</span>
            <input className="input" inputMode="decimal" value={form.Time ?? ""} onChange={(e) => set("Time", e.target.value)} /></label>
          <label className="field">Amount <span className="h">transaction amount in $</span>
            <input className="input" inputMode="decimal" value={form.Amount ?? ""} onChange={(e) => set("Amount", e.target.value)} /></label>
        </div>
        <div style={{ marginTop: 14 }}>
          <button className="disclose" onClick={() => setOpen(!open)} aria-expanded={open}>
            <span>PCA components V1–V28 <span className="cap" style={{ fontWeight: 500 }}>· anonymised by the data provider</span></span>
            <ChevronDown size={18} style={{ transform: open ? "rotate(180deg)" : "none", transition: "transform .2s" }} />
          </button>
          {open && (
            <div className="pca-grid fade" style={{ marginTop: 12 }}>
              {PCA.map((v) => (
                <label key={v} className="field" style={{ fontSize: 11.5 }}>{v}
                  <input className="input sm" inputMode="decimal" value={form[v] ?? ""} onChange={(e) => set(v, e.target.value)} /></label>
              ))}
            </div>
          )}
        </div>
        <button className="btn dark block" style={{ marginTop: 20 }} onClick={predict} disabled={!engine || busy}>
          {busy || !engine ? <Loader2 size={18} className="spin" /> : <Sparkles size={18} />} {engine ? "Predict" : "Loading models…"}
        </button>
      </section>

      <section>
        {!res ? (
          <div className="empty">
            <div className="big"><ScanSearch size={28} /></div>
            <div style={{ fontWeight: 700, color: "var(--ink)", fontSize: 16 }}>No prediction yet</div>
            <div style={{ marginTop: 6 }}>Choose an example on the left and press <b>Predict</b>.</div>
          </div>
        ) : <ResultView res={res} showReport={showReport} setShowReport={setShowReport} />}
      </section>
    </div>
  );
}

function ResultView({ res, showReport, setShowReport }: { res: Result; showReport: boolean; setShowReport: (b: boolean) => void }) {
  const { engine } = useApp();
  const I = ICON[res.verdict];
  const report = engine!.report(res);
  const max = Math.max(...res.factors.map((f) => Math.abs(f.shap)), 1e-9);
  return (
    <div className="fade" style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <div className="verdict" style={{ background: BG[res.verdict] }}>
        <span className="ic" style={{ color: VERDICT_COLOR[res.verdict] }}><I size={26} /></span>
        <div><div className="t">{res.verdict}</div><div className="r">{res.recommendation}</div></div>
      </div>

      {!res.valid ? (
        <div className="alert bad" style={{ display: "block" }}>
          <b>Please fix the following:</b>
          <ul className="list" style={{ marginTop: 6 }}>{res.errors.map((e) => <li key={e}>{e}</li>)}</ul>
        </div>
      ) : (
        <>
          <div className="grid g-3" style={{ gap: 14 }}>
            <Prob title="DNN fraud probability" v={res.dnn_prob!} t={engine!.m.threshold} cls="lav" />
            <Prob title="LR fraud probability" v={res.lr_prob!} t={engine!.m.lr_threshold} cls="blue" />
            <div className={`card stat mini ${res.trust! >= 1 ? "mint" : "peach"}`}>
              <div className="val">{res.trust!.toFixed(2)}</div>
              <div className="lbl">Trust score</div>
              <div className="cap" style={{ marginTop: 14, fontWeight: 600 }}>{res.trust! >= 1 ? "Familiar transaction" : "Unfamiliar · needs ≥ 1.0"}</div>
            </div>
          </div>
          <div className="alert info">
            <span>Models <b>{res.models_agree ? "agree" : "disagree"}</b>: DNN {res.dnn_flag ? "flags" : "clears"}, LR {res.lr_flag ? "flags" : "clears"} this transaction.</span>
          </div>
          <div className="card white">
            <div className="sec-head" style={{ marginBottom: 16 }}>
              <h3>Top 5 factors</h3>
              <span className="hint"><span className="neg">■</span> raises risk · <span className="pos">■</span> lowers risk</span>
            </div>
            <div className="bars">
              {res.factors.map((f) => (
                <div key={f.feature} className="bar-row">
                  <span title={f.label}>{f.feature} <span className="cap">= {Number(f.value.toPrecision(4))}</span></span>
                  <div className="track">
                    <div className="fill" style={{
                      background: f.shap > 0 ? "var(--red)" : "var(--green)", width: `${(Math.abs(f.shap) / max) * 50}%`,
                      left: f.shap > 0 ? "50%" : `${50 - (Math.abs(f.shap) / max) * 50}%`,
                    }} />
                  </div>
                  <span className="num" style={{ textAlign: "right", fontVariantNumeric: "tabular-nums" }}>{f.shap >= 0 ? "+" : ""}{f.shap.toFixed(4)}</span>
                </div>
              ))}
            </div>
            <div className="cap" style={{ marginTop: 14 }}>SHAP values for the DNN (permutation estimate, 100 training background rows).</div>
          </div>
        </>
      )}

      <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
        <button className="btn ghost" onClick={() => setShowReport(!showReport)}><FileText size={16} /> {showReport ? "Hide" : "Show"} full report</button>
        <button className="btn ghost" onClick={() => download("fraud_report.txt", report)}><Download size={16} /> Download .txt</button>
      </div>
      {showReport && <pre className="report fade">{report}</pre>}
    </div>
  );
}

function Prob({ title, v, t, cls }: { title: string; v: number; t: number; cls: string }) {
  return (
    <div className={`card stat mini ${cls}`}>
      <div className={`val ${v >= t ? "neg" : ""}`}>{pct(v)}</div>
      <div className="lbl">{title}</div>
      <div className="meter" title={`threshold ${t}`}><b style={{ width: `${Math.max(v * 100, 1.5)}%` }} /><em style={{ left: `${t * 100}%` }} /></div>
      <div className="cap" style={{ marginTop: 8, fontWeight: 600 }}>threshold {t.toFixed(2)}</div>
    </div>
  );
}
