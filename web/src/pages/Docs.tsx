import { useEffect, useState } from "react";
import { marked } from "marked";
import { Loader2 } from "lucide-react";

const LIMITATIONS = `# Limitations
- **Few positive examples:** only 473 frauds in total (71 in the test set), so recall, precision and subgroup estimates have wide confidence intervals.
- **Anonymised features:** V1–V28 are PCA components; SHAP factors cannot be translated into business reasons for a customer or analyst.
- **Old, short time window:** two days of European card traffic from September 2013; patterns will have drifted since.
- **No customer attributes:** fairness across protected groups cannot be audited; only transaction segments are compared.
- **Random split:** train and test come from the same two days, so the drift baseline and metrics are optimistic about future performance.
- **Decision support only:** the system recommends; humans own MANUAL REVIEW decisions.
- **SHAP is approximate:** the Python service uses KernelExplainer and this web app a permutation estimate, both on 100 background rows, and both explain the DNN only.
- **In-browser scoring:** this site runs the exported DNN, LR and trust index in JavaScript; probabilities match the Python pipeline to within 1e-6.`;

export default function Docs() {
  const [list, setList] = useState<string[]>([]);
  const [cur, setCur] = useState("Limitations");
  const [html, setHtml] = useState("");

  useEffect(() => {
    fetch("/data/docs.json").then((r) => r.json()).then((l: string[]) => { setList(l); setCur(l.includes("FINAL_REPORT.md") ? "FINAL_REPORT.md" : l[0]); });
  }, []);
  useEffect(() => {
    const render = (md: string) => setHtml(String(marked.parse(md.replace(/\]\((?!https?:|#|\/)/g, "](/docs/"))));
    if (cur === "Limitations") render(LIMITATIONS);
    else { setHtml(""); fetch(`/docs/${cur}`).then((r) => r.text()).then(render); }
  }, [cur]);

  const label = (n: string) => n.replace(/\.md$/, "").replace(/_/g, " ").toLowerCase().replace(/\b(\w)/g, (c) => c.toUpperCase())
    .replace(/\b(Srs|Sdd|Ai)\b/g, (w) => w.toUpperCase());

  return (
    <div className="docs">
      <nav className="doc-nav">
        {[...list, "Limitations"].map((n) => <button key={n} className={cur === n ? "on" : ""} onClick={() => setCur(n)}>{label(n)}</button>)}
      </nav>
      <article className="card white" style={{ padding: "28px 32px" }}>
        {html ? <div className="md fade" key={cur} dangerouslySetInnerHTML={{ __html: html }} /> : <Loader2 className="spin" />}
      </article>
    </div>
  );
}
