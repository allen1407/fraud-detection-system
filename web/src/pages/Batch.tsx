import { useRef, useState } from "react";
import Papa from "papaparse";
import { Download, FileSpreadsheet, Loader2, UploadCloud } from "lucide-react";
import { download, pct, useApp, VERDICT_COLOR } from "../data";
import type { Rec, Verdict } from "../engine";
import { VerdictChip } from "./Overview";

type Out = Rec & { verdict: Verdict; dnn_prob: number | null; lr_prob: number | null; models_agree: boolean | null;
  trust: number | null; recommendation: string; errors: string; top_factors: string };
const LIMIT = 5000;

export default function Batch() {
  const { engine } = useApp();
  const [explain, setExplain] = useState(false);
  const [over, setOver] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [out, setOut] = useState<Out[] | null>(null);
  const [cols, setCols] = useState<string[]>([]);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const run = (file: File) => {
    if (!engine) return;
    setOut(null); setMsg(null);
    Papa.parse<Rec>(file, {
      header: true, skipEmptyLines: true,
      complete: async ({ data, meta }) => {
        const missing = engine.m.inputs.filter((c) => !meta.fields?.includes(c));
        if (missing.length) { setMsg({ kind: "bad", text: `The file is missing required columns: ${missing.join(", ")}` }); return; }
        let rows = data;
        if (rows.length > LIMIT) { rows = rows.slice(0, LIMIT); setMsg({ kind: "warn", text: `Only the first ${LIMIT.toLocaleString()} rows are scored.` }); }
        const res: Out[] = [];
        for (let i = 0; i < rows.length; i++) {
          let r;
          try { r = engine.assess(rows[i], explain); } catch (e) {
            r = { valid: false, errors: [`Row could not be validated: ${e}`], verdict: "TRANSACTION REJECTED" as Verdict, recommendation: engine.m.recommendations["TRANSACTION REJECTED"], factors: [] };
          }
          res.push({ ...rows[i], verdict: r.verdict, dnn_prob: r.valid ? r.dnn_prob! : null, lr_prob: r.valid ? r.lr_prob! : null,
            models_agree: r.valid ? r.models_agree! : null, trust: r.valid ? r.trust! : null, recommendation: r.recommendation,
            errors: r.errors.join("; "), top_factors: r.factors.map((f) => `${f.label} (${f.shap >= 0 ? "+" : ""}${Number(f.shap.toPrecision(3))})`).join("; ") });
          if (i % 25 === 24) { setProgress((i + 1) / rows.length); await new Promise((ok) => setTimeout(ok)); }
        }
        setProgress(null); setCols(meta.fields ?? []); setOut(res);
      },
      error: () => setMsg({ kind: "bad", text: "That file could not be read as CSV." }),
    });
  };

  const sampleCsv = () => fetch("/data/sample_batch.csv").then((r) => r.text());
  const counts = out ? out.reduce<Record<string, number>>((a, r) => ({ ...a, [r.verdict]: (a[r.verdict] ?? 0) + 1 }), {}) : {};
  const rejected = counts["TRANSACTION REJECTED"] ?? 0;

  return (
    <>
      <div className="grid g-wide">
        <div
          className={`drop ${over ? "over" : ""}`} onClick={() => fileRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
          onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) run(f); }}>
          <input ref={fileRef} type="file" accept=".csv,text/csv" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) run(f); e.target.value = ""; }} />
          <div className="empty" style={{ border: 0, padding: 0 }}>
            <div className="big"><UploadCloud size={28} /></div>
            <div style={{ fontWeight: 800, color: "var(--ink)", fontSize: 17 }}>Drop a CSV here or click to browse</div>
            <div style={{ marginTop: 6 }}>Up to {LIMIT.toLocaleString()} rows · scored locally, nothing is uploaded</div>
          </div>
        </div>
        <div className="card cream" style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <h3>Need a file?</h3>
          <button className="btn ghost" onClick={async () => download("batch_template.csv", (await sampleCsv()).split("\n").slice(0, 2).join("\n") + "\n", "text/csv")}>
            <Download size={16} /> CSV template</button>
          <button className="btn ghost" onClick={async () => download("sample_batch.csv", await sampleCsv(), "text/csv")}>
            <FileSpreadsheet size={16} /> 20-row sample (2 invalid)</button>
          <button className="btn dark" onClick={async () => run(new File([await sampleCsv()], "sample_batch.csv"))} disabled={!engine}>
            Try the sample now</button>
          <span role="switch" aria-checked={explain} tabIndex={0} className={`toggle ${explain ? "on" : ""}`}
            onClick={() => setExplain(!explain)} onKeyDown={(e) => e.key === " " && setExplain(!explain)}>
            <span className="sw" /> Include SHAP top-5 factors (slower)</span>
        </div>
      </div>
      <div className="cap" style={{ marginTop: 12 }}>Required columns: {engine?.m.inputs.join(", ") ?? "…"}</div>

      {progress !== null && (
        <div className="card white section" style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <Loader2 className="spin" size={18} /><div className="progress" style={{ flex: 1 }}><b style={{ width: `${progress * 100}%` }} /></div>
          <b>{pct(progress, 0)}</b>
        </div>
      )}
      {msg && <div className={`alert ${msg.kind} section`}>{msg.text}</div>}

      {out && (
        <div className="section fade">
          <div className="sec-head">
            <h2>{out.length.toLocaleString()} rows scored</h2>
            <button className="btn dark" style={{ height: 40 }} onClick={() => download("batch_results.csv", Papa.unparse(out), "text/csv")}>
              <Download size={16} /> Download results</button>
          </div>
          <div className="grid g-4">
            {(["AUTO-APPROVE", "MANUAL REVIEW", "AUTO-BLOCK", "TRANSACTION REJECTED"] as Verdict[]).map((v) => (
              <div key={v} className="card mini stat" style={{ background: "#fff", border: "1px solid var(--line)" }}>
                <VerdictChip v={v} />
                <div className="val" style={{ marginTop: 10 }}>{counts[v] ?? 0}</div>
                <div className="progress" style={{ marginTop: 10 }}><b style={{ width: `${((counts[v] ?? 0) / out.length) * 100}%`, background: VERDICT_COLOR[v] }} /></div>
              </div>
            ))}
          </div>
          {rejected > 0 && <div className="alert warn" style={{ marginTop: 16 }}>{rejected} row(s) were rejected; see the errors column for reasons.</div>}
          <div className="card white" style={{ marginTop: 18, padding: "18px 0 6px 18px" }}>
            <div className="tbl-wrap" style={{ maxHeight: 520, overflow: "auto" }}>
              <table className="tbl lined">
                <thead><tr><th>#</th><th>Verdict</th><th className="num">DNN</th><th className="num">LR</th><th className="num">Trust</th><th>Agree</th><th>Errors</th>
                  {explain && <th>Top factors</th>}<th className="num">Amount</th><th className="num">Time</th>{cols.filter((c) => /^V\d+$/.test(c)).map((c) => <th key={c} className="num">{c}</th>)}</tr></thead>
                <tbody>
                  {out.slice(0, 500).map((r, i) => (
                    <tr key={i}>
                      <td className="cap">{i + 1}</td><td><VerdictChip v={r.verdict} /></td>
                      <td className="num">{r.dnn_prob == null ? "–" : pct(r.dnn_prob)}</td>
                      <td className="num">{r.lr_prob == null ? "–" : pct(r.lr_prob)}</td>
                      <td className="num">{r.trust == null ? "–" : r.trust.toFixed(2)}</td>
                      <td>{r.models_agree == null ? "–" : r.models_agree ? "Yes" : "No"}</td>
                      <td style={{ color: "var(--red)", fontWeight: 500 }}>{r.errors}</td>
                      {explain && <td style={{ fontWeight: 500 }}>{r.top_factors}</td>}
                      <td className="num">{String(r.Amount)}</td><td className="num">{String(r.Time)}</td>
                      {cols.filter((c) => /^V\d+$/.test(c)).map((c) => <td key={c} className="num" style={{ fontWeight: 500 }}>{Number(r[c]).toFixed(3)}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          {out.length > 500 && <div className="cap" style={{ marginTop: 8 }}>Showing the first 500 rows; the download contains all of them.</div>}
        </div>
      )}
    </>
  );
}
