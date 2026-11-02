import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CheckCircle2, RefreshCcw, RotateCcw, Radar, SlidersHorizontal, XCircle } from "lucide-react";
import { useApp } from "../data";

const axis = { tick: { fontSize: 11, fill: "#71717a" }, stroke: "#e4e4e7" };

export default function Monitoring() {
  const { reports: R } = useApp();
  const fails = R.readiness.filter((r) => r.status === "FAIL");
  const passed = R.readiness.length - fails.length;
  const maxPsi = Math.max(...R.drift.map((d) => d.psi));

  return (
    <>
      <div className="grid g-wide">
        <div className="card white">
          <div className="sec-head" style={{ marginBottom: 4 }}>
            <h3>Population Stability Index · train vs test</h3>
            <span className="hint">max {maxPsi.toFixed(4)}</span>
          </div>
          <div style={{ height: 640 }}>
            <ResponsiveContainer>
              <BarChart data={R.drift} layout="vertical" margin={{ top: 18, right: 20, bottom: 0, left: 0 }}>
                <CartesianGrid stroke="#f1f1f4" horizontal={false} />
                <XAxis type="number" domain={[0, 0.3]} {...axis} />
                <YAxis type="category" dataKey="feature" {...axis} width={78} interval={0} />
                <Tooltip formatter={(v: number) => v.toFixed(5)} cursor={{ fill: "#f6f7fb" }} />
                <ReferenceLine x={0.1} stroke="#d97706" strokeDasharray="4 4" label={{ value: "watch 0.10", position: "top", fontSize: 11, fill: "#d97706" }} />
                <ReferenceLine x={0.25} stroke="#e5484d" strokeDasharray="4 4" label={{ value: "retrain 0.25", position: "top", fontSize: 11, fill: "#e5484d" }} />
                <Bar isAnimationActive={false} dataKey="psi" radius={[0, 6, 6, 0]} barSize={10} minPointSize={2}>
                  {R.drift.map((d) => <Cell key={d.feature} fill={d.psi > 0.25 ? "#e5484d" : d.psi > 0.1 ? "#d97706" : "#2563eb"} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
          <div className={`card stat ${fails.length ? "peach" : "mint"}`}>
            <div className="val">{passed}/{R.readiness.length}</div>
            <div className="lbl">Readiness checks passing</div>
            <div className="foot"><span className="tile">{fails.length ? <XCircle size={20} /> : <CheckCircle2 size={20} />}</span>
              <span className="delta">{fails.length ? `${fails.length} to resolve` : "Ready"}</span></div>
          </div>
          <div className="card blue">
            <h3>Retraining & rollback policy</h3>
            <div style={{ display: "flex", flexDirection: "column", gap: 14, marginTop: 14 }}>
              {[[Radar, "Monitoring", "Compute PSI per numeric feature monthly against the training distribution."],
                [RefreshCcw, "Retrain trigger", "Any feature PSI > 0.25, or quarterly, whichever comes first."],
                [RotateCcw, "Rollback", "Keep model + preprocessor + metadata together in a dated folder; restore the previous set together if checks fail."],
                [SlidersHorizontal, "Recalibration", "Data covers two days in Sept 2013; re-validate thresholds and costs on recent traffic before real use."]]
                .map(([I, h, p]) => {
                  const Icon = I as typeof Radar;
                  return (
                    <div key={h as string} style={{ display: "flex", gap: 12 }}>
                      <span className="tile" style={{ width: 36, height: 36, borderRadius: 11, background: "#fff", display: "grid", placeItems: "center", flex: "none" }}><Icon size={17} /></span>
                      <div><b style={{ fontSize: 14 }}>{h as string}</b><div className="cap" style={{ color: "var(--ink-2)", lineHeight: 1.5 }}>{p as string}</div></div>
                    </div>
                  );
                })}
            </div>
          </div>
        </div>
      </div>

      <section className="section">
        <div className="sec-head"><h2>Deployment readiness checklist</h2><span className="hint">Measured values</span></div>
        {fails.length > 0 && <div className="alert bad" style={{ marginBottom: 14 }}>{fails.length} of {R.readiness.length} checks fail: {fails.map((f) => f.check).join(", ")}. Resolve or formally accept these before production use.</div>}
        <div className="card white" style={{ padding: "18px 0 6px 18px" }}>
          <div className="tbl-wrap">
            <table className="tbl lined">
              <thead><tr><th>Check</th><th>Criterion</th><th>Measured</th><th>Status</th></tr></thead>
              <tbody>{R.readiness.map((r) => (
                <tr key={r.check}><td>{r.check}</td><td style={{ fontWeight: 500, color: "var(--ink-2)" }}>{r.criterion}</td><td>{r.measured}</td>
                  <td><span className={`chip ${r.status === "PASS" ? "v-approve" : r.status === "FAIL" ? "v-block" : "v-reject"}`}><i />{r.status}</span></td></tr>
              ))}</tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="sec-head"><h2>Automated tests</h2><span className="hint">pytest</span></div>
        <div className="grid g-3">
          {R.testing.map((t) => (
            <div key={t.test} className="card mini" style={{ display: "flex", gap: 12, alignItems: "center" }}>
              {t.result === "PASSED" ? <CheckCircle2 size={20} className="pos" /> : <XCircle size={20} className="neg" />}
              <div style={{ minWidth: 0 }}><div style={{ fontWeight: 700, fontSize: 13, overflow: "hidden", textOverflow: "ellipsis" }}>{t.test.replace(/^test_/, "").replace(/_/g, " ")}</div>
                <div className="cap">{t.result.toLowerCase()} · {t.duration_s}s</div></div>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
