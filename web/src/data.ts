import { createContext, useContext } from "react";
import type { Engine, Rec, Verdict } from "./engine";

type Row = Record<string, string | number | boolean | null>;
type Conf = { tn: number; fp: number; fn: number; tp: number };
type Scores = { threshold: number; roc_auc: number; pr_auc: number; f1: number; precision: number; recall: number;
  accuracy: number; flag_rate: number; confusion_matrix: Conf };

export interface Reports {
  metrics: { n_test: number; test_fraud_rate: number; dnn: Scores; lr: Scores; dnn_at_cost_optimal: Scores;
    thresholds: Record<string, number>; routing: Record<string, number> };
  curves: Record<"dnn" | "lr", { roc: [number, number][]; pr: [number, number][] }>;
  sweep: (Conf & { t: number })[];
  traffic: { h: number; n: number; fraud: number }[];
  cost: { approve_all_cost: number; flag_all_cost: number; optimum_threshold: number; optimum_cost: number };
  cost_fn: number; cost_fp: number;
  latency: { single_no_explanation_ms: { p50: number; p95: number }; single_with_shap_ms: { p50: number; p95: number };
    batch: { rows_per_second: number } };
  data: { rows: number; fraud_rows: number; fraud_rate: number; time_span_hours: number; duplicates_removed: number; raw_rows: number; amount_max: number };
  n_features: number;
  fairness_amount: Row[]; fairness_time: Row[]; robustness: Row[];
  robustness_summary: { dnn_flip_rate: number; verdict_flip_rate: number; no_flip_cases: number; n_cases: number; search_max_std: number };
  shap_stability: { mean_spearman: number; std_spearman: number; min_spearman: number; n_samples: number };
  shap_global: { feature: string; mean_abs_shap: number }[];
  drift: { feature: string; psi: number; status: string }[];
  readiness: { check: string; criterion: string; measured: string; status: string }[];
  testing: { test: string; file: string; result: string; duration_s: number }[];
}
export interface Samples {
  presets: Record<string, Rec>;
  recent: { id: string; record: Rec; label: number; dnn_prob: number; lr_prob: number; trust: number; verdict: Verdict }[];
}

export interface Ctx {
  reports: Reports; samples: Samples; engine: Engine | null;
  go: (page: string) => void; loadIntoScorer: (rec: Rec) => void; pending: Rec | null; clearPending: () => void;
}
export const AppCtx = createContext<Ctx>(null!);
export const useApp = () => useContext(AppCtx);

export const VERDICT_CLASS: Record<Verdict, string> = {
  "AUTO-BLOCK": "v-block", "AUTO-APPROVE": "v-approve", "MANUAL REVIEW": "v-review", "TRANSACTION REJECTED": "v-reject",
};
export const VERDICT_COLOR: Record<Verdict, string> = {
  "AUTO-BLOCK": "#e5484d", "AUTO-APPROVE": "#16a34a", "MANUAL REVIEW": "#d97706", "TRANSACTION REJECTED": "#71717a",
};

export const pct = (v: number, d = 2) => `${(v * 100).toFixed(d)}%`;
export const fmt = (v: number, d = 3) => v.toFixed(d);
export const int = (v: number) => Math.round(v).toLocaleString("en-US");
export const money = (v: number) => `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
export const hhmm = (sec: number) => {
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60);
  return `Day ${Math.floor(h / 24) + 1} · ${String(h % 24).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
};

export function download(name: string, text: string, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
