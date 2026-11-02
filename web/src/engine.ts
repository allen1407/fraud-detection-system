// In-browser port of src/pipeline.py: validate -> engineer -> score (DNN + LR) -> trust -> route -> explain -> report.
export type Verdict = "AUTO-BLOCK" | "AUTO-APPROVE" | "MANUAL REVIEW" | "TRANSACTION REJECTED";
export type Rec = Record<string, unknown>;

interface Layer { W: number[][]; b: number[]; act: string }
export interface ModelJson {
  features: string[]; labels: Record<string, string>; inputs: string[];
  mean: number[]; scale: number[]; dnn: Layer[]; lr: { coef: number[]; intercept: number };
  threshold: number; lr_threshold: number; trust_min: number; knn_k: number; ref_distance: number;
  ranges: Record<string, [number, number]>; recommendations: Record<Verdict, string>;
  shap_background: number[][]; trained_at: string;
}
export interface Factor { feature: string; label: string; value: number; shap: number; direction: string }
export interface Result {
  valid: boolean; errors: string[]; verdict: Verdict; recommendation: string; record: Rec;
  dnn_prob?: number; lr_prob?: number; dnn_flag?: boolean; lr_flag?: boolean; models_agree?: boolean;
  trust?: number; factors: Factor[];
}

const TIME = "Time";
const g = (v: number) => String(Number(v.toPrecision(6)));       // Python's {:g}
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export class Engine {
  m: ModelJson; ref: Float32Array; nRef: number; d: number;
  private bufs: Float64Array[];
  private bgStd: Float64Array[];

  constructor(m: ModelJson, ref: Float32Array) {
    this.m = m; this.ref = ref; this.d = m.mean.length; this.nRef = ref.length / this.d;
    this.bufs = m.dnn.map((l) => new Float64Array(l.b.length));
    this.bgStd = m.shap_background.map((r) => this.standardize(r));
  }

  static async load(base = "/data") {
    const [m, buf] = await Promise.all([
      fetch(`${base}/model.json`).then((r) => r.json()),
      fetch(`${base}/reference.bin`).then((r) => r.arrayBuffer()),
    ]);
    return new Engine(m, new Float32Array(buf));
  }

  validate(rec: Rec): string[] {
    const errors: string[] = [];
    const missing = (v: unknown) => v === null || v === undefined || (typeof v === "number" && Number.isNaN(v)) ||
      (typeof v === "string" && !v.trim());
    for (const f of this.m.inputs) if (missing(rec[f])) errors.push(`Missing required field: ${f}`);
    if (errors.length) return errors;
    for (const f of this.m.inputs) {
      const raw = rec[f];
      const v = typeof raw === "number" ? raw : typeof raw === "string" ? Number(raw.trim()) : NaN;
      if (typeof raw === "string" && /^[+-]?(inf(inity)?|nan)$/i.test(raw.trim())) {
        errors.push(`${f} must be a finite number`); continue;
      }
      if (Number.isNaN(v)) { errors.push(`${f} must be a number`); continue; }
      if (!Number.isFinite(v)) errors.push(`${f} must be a finite number`);
      else if ((f === "Amount" || f === TIME) && v < 0) errors.push(`${f} must not be negative`);
      else {
        const [lo, hi] = this.m.ranges[f];
        if (!(lo <= v && v <= hi)) errors.push(`${f}=${g(v)} is outside the valid range [${g(lo)}, ${g(hi)}]`);
      }
    }
    return errors;
  }

  /** Feature engineering: V1-V28, Amount, hour, is_night, log_amount (raw Time is not a feature). */
  featurize(rec: Rec): number[] {
    const n = (f: string) => Number(rec[f]);
    const hour = Math.floor(n(TIME) / 3600) % 24;
    return this.m.features.map((f) =>
      f === "hour" ? hour : f === "is_night" ? (hour < 6 ? 1 : 0) : f === "log_amount" ? Math.log1p(n("Amount")) : n(f));
  }

  standardize(x: number[]): Float64Array {
    const z = new Float64Array(this.d);
    for (let i = 0; i < this.d; i++) z[i] = (x[i] - this.m.mean[i]) / this.m.scale[i];
    return z;
  }

  dnn(z: ArrayLike<number>): number {
    let inp = z;
    this.m.dnn.forEach((l, k) => {
      const out = this.bufs[k], nOut = l.b.length;
      for (let j = 0; j < nOut; j++) out[j] = l.b[j];
      for (let i = 0; i < inp.length; i++) {
        const xi = inp[i], Wi = l.W[i];
        if (xi !== 0) for (let j = 0; j < nOut; j++) out[j] += xi * Wi[j];
      }
      for (let j = 0; j < nOut; j++) out[j] = l.act === "relu" ? Math.max(0, out[j]) : l.act === "sigmoid" ? sigmoid(out[j]) : out[j];
      inp = out;
    });
    return inp[0];
  }

  lr(z: ArrayLike<number>): number {
    let s = this.m.lr.intercept;
    for (let i = 0; i < this.d; i++) s += this.m.lr.coef[i] * z[i];
    return sigmoid(s);
  }

  /** Familiarity: 95th-pct reference kNN distance / this row's mean kNN distance, capped at 10. */
  trust(z: ArrayLike<number>): number {
    const k = this.m.knn_k, best = new Array(k).fill(Infinity), d = this.d, ref = this.ref;
    for (let r = 0; r < this.nRef; r++) {
      let s = 0;
      const o = r * d;
      for (let i = 0; i < d && s < best[k - 1]; i++) { const t = z[i] - ref[o + i]; s += t * t; }
      if (s < best[k - 1]) {
        let p = k - 1;
        while (p > 0 && best[p - 1] > s) { best[p] = best[p - 1]; p--; }
        best[p] = s;
      }
    }
    const mean = best.reduce((a, b) => a + Math.sqrt(b), 0) / k;
    return Math.min(this.m.ref_distance / Math.max(mean, 1e-9), 10);
  }

  route(dnnP: number, lrP: number, trust: number): Verdict {
    const df = dnnP >= this.m.threshold, lf = lrP >= this.m.lr_threshold, ok = trust >= this.m.trust_min;
    if (ok && df && lf) return "AUTO-BLOCK";
    if (ok && !df && !lf) return "AUTO-APPROVE";
    return "MANUAL REVIEW";
  }

  /** Permutation SHAP (antithetic orderings) for the DNN against the 100-row training background. */
  shap(x: number[], perms = 6): number[] {
    const d = this.d, z = this.standardize(x), phi = new Float64Array(d), cur = new Float64Array(d);
    let seed = 42;
    const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
    let count = 0;
    for (let p = 0; p < perms; p++) {
      const order = Array.from({ length: d }, (_, i) => i);
      for (let i = d - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
      for (const ord of [order, [...order].reverse()]) {
        for (const bg of this.bgStd) {
          cur.set(bg);
          let prev = this.dnn(cur);
          for (const j of ord) { cur[j] = z[j]; const v = this.dnn(cur); phi[j] += v - prev; prev = v; }
          count++;
        }
      }
    }
    return Array.from(phi, (v) => v / count);
  }

  explain(rec: Rec, topK = 5): Factor[] {
    const x = this.featurize(rec), sv = this.shap(x);
    return sv.map((s, i) => i).sort((a, b) => Math.abs(sv[b]) - Math.abs(sv[a])).slice(0, topK).map((i) => ({
      feature: this.m.features[i], label: this.m.labels[this.m.features[i]], value: x[i], shap: sv[i],
      direction: sv[i] > 0 ? "increases fraud risk" : "decreases fraud risk",
    }));
  }

  assess(rec: Rec, explain = true): Result {
    const errors = this.validate(rec);
    if (errors.length)
      return { valid: false, errors, verdict: "TRANSACTION REJECTED", recommendation: this.m.recommendations["TRANSACTION REJECTED"], record: rec, factors: [] };
    const z = this.standardize(this.featurize(rec));
    const dnn_prob = this.dnn(z), lr_prob = this.lr(z), trust = this.trust(z);
    const verdict = this.route(dnn_prob, lr_prob, trust);
    const dnn_flag = dnn_prob >= this.m.threshold, lr_flag = lr_prob >= this.m.lr_threshold;
    return {
      valid: true, errors: [], record: rec, verdict, dnn_prob, lr_prob, dnn_flag, lr_flag,
      models_agree: dnn_flag === lr_flag, trust, recommendation: this.m.recommendations[verdict],
      factors: explain ? this.explain(rec) : [],
    };
  }

  report(r: Result): string {
    const L = ["=".repeat(60), "FRAUD DETECTION REPORT", "=".repeat(60)];
    if (!r.valid) {
      L.push("VERDICT: TRANSACTION REJECTED", "", "Reasons:", ...r.errors.map((e) => `  - ${e}`), "", `Recommendation: ${r.recommendation}`);
      return L.join("\n");
    }
    L.push(`VERDICT: ${r.verdict}`, "",
      `DNN fraud probability : ${r.dnn_prob!.toFixed(3)} (threshold ${this.m.threshold.toFixed(2)}) -> ${r.dnn_flag ? "FLAG" : "OK"}`,
      `LR  fraud probability : ${r.lr_prob!.toFixed(3)} (threshold ${this.m.lr_threshold.toFixed(2)}) -> ${r.lr_flag ? "FLAG" : "OK"}`,
      `Model agreement       : ${r.models_agree ? "AGREE" : "DISAGREE"}`,
      `Trust score           : ${r.trust!.toFixed(2)} (>= ${this.m.trust_min.toFixed(1)} required for automatic action)`,
      "", `Recommendation: ${r.recommendation}`, "", "Top factors (SHAP, DNN):");
    r.factors.forEach((f, i) => L.push(`  ${i + 1}. ${f.label} = ${Number(f.value.toPrecision(4))}: ${f.direction} (${f.shap >= 0 ? "+" : ""}${f.shap.toFixed(4)})`));
    L.push("", "Note: decision support only; final decisions on MANUAL REVIEW cases rest with a human analyst.");
    return L.join("\n");
  }
}
