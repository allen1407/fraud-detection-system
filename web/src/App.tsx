import { useCallback, useEffect, useState } from "react";
import { BookOpen, Gauge, Layers, LayoutGrid, Loader2, Scale, ScanSearch, ShieldCheck, Activity, Bell, CircleHelp } from "lucide-react";
import { Engine, type Rec } from "./engine";
import { AppCtx, type Reports, type Samples } from "./data";
import Overview from "./pages/Overview";
import Score from "./pages/Score";
import Batch from "./pages/Batch";
import Performance from "./pages/Performance";
import Responsible from "./pages/Responsible";
import Monitoring from "./pages/Monitoring";
import Docs from "./pages/Docs";

const PAGES = [
  { id: "overview", title: "Overview", sub: "Intelligent fraud detection · decision support", icon: LayoutGrid, C: Overview },
  { id: "score", title: "Score Transaction", sub: "Pick an example or edit the values, then predict", icon: ScanSearch, C: Score },
  { id: "batch", title: "Batch Scoring", sub: "Upload a CSV and score every row in your browser", icon: Layers, C: Batch },
  { id: "performance", title: "Model Performance", sub: "Held-out test set results for the DNN and LR", icon: Gauge, C: Performance },
  { id: "responsible", title: "Responsible AI", sub: "Fairness, robustness, explainability, human oversight", icon: Scale, C: Responsible },
  { id: "monitoring", title: "Monitoring & Readiness", sub: "Drift baseline, readiness checks, retraining policy", icon: Activity, C: Monitoring },
  { id: "docs", title: "Documentation", sub: "Project reports, manuals and limitations", icon: BookOpen, C: Docs },
];

const pageFromHash = () => {
  const id = location.hash.replace(/^#\/?/, "");
  return PAGES.some((p) => p.id === id) ? id : "overview";
};

export default function App() {
  const [page, setPage] = useState(pageFromHash);
  const [reports, setReports] = useState<Reports | null>(null);
  const [samples, setSamples] = useState<Samples | null>(null);
  const [engine, setEngine] = useState<Engine | null>(null);
  const [pending, setPending] = useState<Rec | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    const on = () => { setPage(pageFromHash()); window.scrollTo({ top: 0 }); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);

  useEffect(() => {
    Promise.all([fetch("/data/reports.json").then((r) => r.json()), fetch("/data/samples.json").then((r) => r.json())])
      .then(([r, s]) => { setReports(r); setSamples(s); })
      .catch(() => setError(true));
    Engine.load().then(setEngine).catch(() => setError(true));
  }, []);

  const go = useCallback((p: string) => { location.hash = `/${p}`; }, []);
  const loadIntoScorer = useCallback((rec: Rec) => { setPending(rec); location.hash = "/score"; }, []);
  const meta = PAGES.find((p) => p.id === page)!;

  return (
    <div className="frame">
      <div className="shell">
        <aside className="rail">
          <a className="logo" href="#/overview" aria-label="Home"><ShieldCheck size={22} strokeWidth={2.4} /></a>
          <nav className="nav">
            {PAGES.map(({ id, title, icon: I }) => (
              <a key={id} href={`#/${id}`} className={page === id ? "on" : ""} aria-label={title}>
                <I size={20} /><span className="tip">{title}</span>
              </a>
            ))}
          </nav>
          <div className="nav bottom"><a href="#/docs" aria-label="Help"><CircleHelp size={20} /><span className="tip">Help</span></a></div>
        </aside>

        <main className="panel">
          <header className="top">
            <div>
              <h1>{meta.title}</h1>
              <div className="sub">{meta.sub}</div>
            </div>
            <div className="top-right">
              <span className="status" title="Models run locally in your browser; no data leaves this page">
                <span className={`dot ${engine ? "" : "wait"}`} />
                <span className="txt">{engine ? "Models ready · in-browser" : "Loading models…"}</span>
              </span>
              <button className="icon-btn" aria-label="Notifications"><Bell size={17} /></button>
              <span className="user"><span className="avatar">FA</span><span className="uname">Fraud Analyst</span></span>
            </div>
          </header>

          {error ? (
            <div className="alert bad">The model and report files could not be loaded. Refresh the page to try again.</div>
          ) : !reports || !samples ? (
            <div className="empty"><Loader2 className="spin" /> <div>Loading…</div></div>
          ) : (
            <AppCtx.Provider value={{ reports, samples, engine, go, loadIntoScorer, pending, clearPending: () => setPending(null) }}>
              <div className="fade" key={page}><meta.C /></div>
            </AppCtx.Provider>
          )}
        </main>
      </div>
    </div>
  );
}
