"""Task 10 evaluation: every number in reports/ is computed here from the real saved models."""
import json
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)

from . import config as C
from .features import NUMERIC_FEATURES
from .pipeline import FraudPipeline

R, F = C.REPORTS_DIR, C.FIGURES_DIR
BLUE, ORANGE, GREY, RED = "#2563eb", "#ea580c", "#6b7280", "#dc2626"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})


def cm(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def model_metrics(y, p, t):
    pred = (p >= t).astype(int)
    return dict(threshold=float(t), roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
                f1=float(f1_score(y, pred)), precision=float(precision_score(y, pred, zero_division=0)),
                recall=float(recall_score(y, pred)), accuracy=float(accuracy_score(y, pred)),
                flag_rate=float(pred.mean()), confusion_matrix=cm(y, pred))


def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def save_fig(name):
    plt.tight_layout(); plt.savefig(F / name); plt.close()


def predict(pipe, df):
    s = pipe.score(df[C.INPUT_COLUMNS])
    X = np.vstack(s["_X"].to_numpy())
    return s.dnn_prob.to_numpy(), s.lr_prob.to_numpy(), X


# ---------------------------------------------------------------------------------------------
def eval_metrics(pipe, test):
    y = test[C.TARGET].values
    dp, lp, X = predict(pipe, test)
    meta = pipe.meta
    out = dict(n_test=int(len(y)), test_fraud_rate=float(y.mean()),
               dnn=model_metrics(y, dp, pipe.threshold), lr=model_metrics(y, lp, pipe.lr_threshold),
               thresholds=dict(dnn_operating=pipe.threshold, lr=pipe.lr_threshold,
                               dnn_f1_unconstrained=meta["threshold_f1_unconstrained"],
                               dnn_f1_guardrail=meta["threshold_f1_guardrail"],
                               dnn_cost_optimal=meta["threshold_cost_optimal"]))
    out["dnn_at_cost_optimal"] = model_metrics(y, dp, meta["threshold_cost_optimal"])
    trust = pipe.trust_scores(X)
    verdicts = [pipe.route(a, b, c) for a, b, c in zip(dp, lp, trust)]
    out["routing"] = {v: int(n) for v, n in pd.Series(verdicts).value_counts().items()}
    pd.DataFrame(dict(y_true=y, dnn_prob=dp, lr_prob=lp, trust=trust, verdict=verdicts)).to_csv(R / "test_predictions.csv", index=False)
    (R / "metrics.json").write_text(json.dumps(out, indent=2))

    plt.figure(figsize=(5, 4.5))
    for name, p, c in [("DNN", dp, BLUE), ("LR", lp, ORANGE)]:
        fpr, tpr, _ = roc_curve(y, p); plt.plot(fpr, tpr, c=c, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "--", c=GREY, label="Chance"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve (test set)"); plt.legend(); save_fig("roc_curve.png")

    plt.figure(figsize=(5, 4.5))
    for name, p, c in [("DNN", dp, BLUE), ("LR", lp, ORANGE)]:
        pr, rc, _ = precision_recall_curve(y, p); plt.plot(rc, pr, c=c, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    plt.axhline(y.mean(), ls="--", c=GREY, label=f"Base rate {y.mean():.4f}"); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-recall curve (test set)"); plt.legend(); save_fig("pr_curve.png")

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, m) in zip(axs, [("DNN", out["dnn"]), ("LR", out["lr"])]):
        c = m["confusion_matrix"]; mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(mat, cmap="Blues"); ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, mat[i, j], ha="center", va="center", fontsize=14, color="white" if mat[i, j] > mat.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Legit", "Fraud"]); ax.set_yticks([0, 1], ["Legit", "Fraud"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{name} @ {m['threshold']:.2f}")
    save_fig("confusion_matrix.png")
    return y, dp, lp, X, trust


def eval_cost(y, dp):
    rows = []
    for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
        c = cm(y, (dp >= t).astype(int))
        rows.append(dict(threshold=float(t), **c, cost=c["fn"] * C.COST_FN + c["fp"] * C.COST_FP,
                         f1=f1_score(y, (dp >= t).astype(int), zero_division=0)))
    df = pd.DataFrame(rows)
    df["is_optimum"] = df.cost == df.cost.min()
    df.to_csv(R / "cost_threshold.csv", index=False)
    approve_all, flag_all = y.sum() * C.COST_FN, (1 - y).sum() * C.COST_FP
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df.threshold, df.cost, c=BLUE, label="Expected cost (DNN)")
    ax.axhline(approve_all, c=GREY, ls="--", label=f"Approve everything (\\${approve_all:,.0f})")
    ax.axhline(flag_all, c=ORANGE, ls="--", label=f"Flag everything (\\${flag_all:,.0f})")
    o = df.loc[df.cost.idxmin()]; ax.scatter([o.threshold], [o.cost], c=RED, zorder=5, label=f"Optimum t={o.threshold:.2f}")
    ax.set_yscale("log")                                     # flag-everything dwarfs the other costs at 0.17% prevalence
    ax.set_xlabel("Threshold"); ax.set_ylabel("Cost on test set (USD, log scale)"); ax.set_title(f"Cost vs threshold (FN=\\${C.COST_FN:.0f}, FP=\\${C.COST_FP:.0f})")
    ax.legend(fontsize=8); save_fig("cost_threshold_analysis.png")
    return dict(approve_all_cost=float(approve_all), flag_all_cost=float(flag_all), optimum_threshold=float(o.threshold),
                optimum_cost=float(o.cost))


def eval_latency(pipe, test):
    recs = test[C.INPUT_COLUMNS].to_dict("records")
    pipe.assess(recs[0]); pipe.assess(recs[0], explain=False)
    res = {}
    for key, expl in [("single_no_explanation_ms", False), ("single_with_shap_ms", True)]:
        ts = []
        for r in recs[:100]:
            t = time.perf_counter(); pipe.assess(r, explain=expl); ts.append((time.perf_counter() - t) * 1000)
        res[key] = dict(mean=float(np.mean(ts)), p50=float(np.percentile(ts, 50)), p95=float(np.percentile(ts, 95)), n=len(ts))
    batch = test[C.INPUT_COLUMNS].head(5000)                 # same cap as the app's batch page
    t = time.perf_counter(); pipe.score_batch(batch); dt = time.perf_counter() - t
    res["batch"] = dict(rows=int(len(batch)), seconds=float(dt), rows_per_second=float(len(batch) / dt))
    (R / "latency.json").write_text(json.dumps(res, indent=2))
    return res


def eval_shap(pipe, test, X_unused=None):
    rng = np.random.default_rng(C.SEED)
    sub = test.sample(150, random_state=C.SEED)
    M = pipe.fe.transform(sub[C.INPUT_COLUMNS]).to_numpy(float)
    np.random.seed(C.SEED); sv = pipe.shap_values(M)
    cols = NUMERIC_FEATURES
    imp = pd.DataFrame(dict(feature=cols, mean_abs_shap=np.abs(sv).mean(0))).sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(R / "shap_global.csv", index=False)
    import shap
    shap.summary_plot(sv, pd.DataFrame(M, columns=cols), show=False, plot_size=(8, 6))
    plt.title("SHAP summary (DNN, 150 test rows)"); save_fig("shap_summary.png")

    # stability: 1% multiplicative Gaussian noise on V1-V28 and Amount, 30 samples
    stab = []
    for _, row in test.sample(30, random_state=1).iterrows():
        r = row[C.INPUT_COLUMNS].to_dict()
        noisy = dict(r)
        for f in C.PERTURB_INPUTS:
            noisy[f] = float(r[f]) * (1 + 0.01 * rng.standard_normal())
        out = []
        for rec in (r, noisy):
            Mi = pipe.fe.transform(pipe._frame([rec])).to_numpy(float)
            np.random.seed(C.SEED); out.append(pipe.shap_values(Mi)[0])
        stab.append(spearmanr(out[0], out[1])[0])
    res = dict(n_samples=30, noise="1% multiplicative Gaussian on V1-V28 and Amount", mean_spearman=float(np.mean(stab)),
               std_spearman=float(np.std(stab)), min_spearman=float(np.min(stab)))
    (R / "shap_stability.json").write_text(json.dumps(res, indent=2))
    return res


SUBGROUPS = [  # (column, file stem, title); the dataset has no customer attributes, so audit transaction segments
    ("amount_band", "fairness_amount", "amount band"),
    ("time_of_day", "fairness_time", "time of day"),
]


def eval_fairness(test, y, dp, thr):
    df = test.copy(); df["pred"] = (dp >= thr).astype(int); df["y"] = y
    df["amount_band"] = pd.cut(df.Amount, [-1, 10, 100, 500, np.inf], labels=["<=$10", "$10-100", "$100-500", ">$500"])
    hour = (df[C.TIME_COL] // 3600) % 24
    df["time_of_day"] = pd.cut(hour, [-1, 5, 17, 23], labels=["Night 00-06", "Day 06-18", "Evening 18-24"])
    out = {}
    for col, stem, title in SUBGROUPS:
        rows = []
        for g, d in df.groupby(col, observed=True):
            tp = ((d.pred == 1) & (d.y == 1)).sum(); fp = ((d.pred == 1) & (d.y == 0)).sum()
            rows.append(dict(group=g, n=len(d), n_fraud=int(d.y.sum()), flag_rate=d.pred.mean(),
                             tpr=tp / max((d.y == 1).sum(), 1), fpr=fp / max((d.y == 0).sum(), 1),
                             precision=tp / max(d.pred.sum(), 1)))
        t = pd.DataFrame(rows)
        gaps = {c: t[c].max() - t[c].min() for c in ["flag_rate", "tpr", "fpr", "precision"]}
        gap_row = dict(group="MAX GAP", n=int(t.n.sum()), n_fraud=int(t.n_fraud.sum()), **gaps)
        pd.concat([t, pd.DataFrame([gap_row])]).to_csv(R / f"{stem}.csv", index=False)
        out[col] = gaps
        fig, ax = plt.subplots(figsize=(7, 3.8)); w = 0.2; x = np.arange(len(t))
        for i, (m, c) in enumerate([("flag_rate", GREY), ("tpr", BLUE), ("fpr", ORANGE), ("precision", "#16a34a")]):
            ax.bar(x + i * w - 1.5 * w, t[m], w, label=m.upper().replace("_", " "), color=c)
        ax.set_xticks(x, [f"{g}\n(n={n}, fraud={k})" for g, n, k in zip(t.group, t.n, t.n_fraud)]); ax.set_ylim(0, 1)
        ax.legend(ncol=4, fontsize=8); ax.set_title(f"Subgroup performance by {title} (DNN, test set)")
        save_fig(f"{stem}.png")
    return out


def eval_robustness(pipe, test, n_cases=15, n_dirs=400, max_mag=3.0):
    rng = np.random.default_rng(C.SEED)
    P = C.PERTURB_INPUTS
    scale = pipe.train[P].std().values
    n_fraud = min(5, int(test[C.TARGET].sum()))
    cases = pd.concat([test[test[C.TARGET] == 1].sample(n_fraud, random_state=C.SEED),
                       test[test[C.TARGET] == 0].sample(n_cases - n_fraud, random_state=C.SEED)])
    rows = []
    for _, row in cases.iterrows():
        r = row[C.INPUT_COLUMNS].to_dict()
        base = pipe.assess(r, explain=False)
        base_flag = base["dnn_prob"] >= pipe.threshold
        x0 = np.array([float(r[f]) for f in P])
        best = None
        mags = np.linspace(0.02, max_mag, 40)
        dirs = rng.standard_normal((n_dirs, len(x0))); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        for mag in mags:                                       # smallest magnitude first
            cands = []
            for d in dirs:
                x = x0 + mag * d * scale
                for k, f in enumerate(P):
                    x[k] = np.clip(x[k], *C.RANGES[f])
                cands.append(dict(r, **{f: float(v) for f, v in zip(P, x)}))
            sc = pipe.score(cands)
            flips = np.where((sc.dnn_prob.to_numpy() >= pipe.threshold) != base_flag)[0]
            if len(flips):
                i = flips[0]; cf = cands[i]
                dist = float(np.linalg.norm((np.array([cf[f] for f in P]) - x0) / scale))
                best = (dist, cf); break
        if best is None:
            rows.append(dict(case=len(rows) + 1, is_fraud=int(row[C.TARGET]), original_prob=base["dnn_prob"], original_verdict=base["verdict"], distance_std=np.nan,
                             final_prob=np.nan, dnn_class_flipped=False, final_verdict=base["verdict"], verdict_flipped=False))
        else:
            res = pipe.assess(best[1], explain=False)
            rows.append(dict(case=len(rows) + 1, is_fraud=int(row[C.TARGET]), original_prob=base["dnn_prob"], original_verdict=base["verdict"], distance_std=best[0],
                             final_prob=res["dnn_prob"], dnn_class_flipped=True, final_verdict=res["verdict"],
                             verdict_flipped=res["verdict"] != base["verdict"]))
    df = pd.DataFrame(rows); df.to_csv(R / "robustness.csv", index=False)
    # cases with no flip inside the search radius count as >= max_mag (censored), so the median stays defined
    return dict(median_distance=float(df.distance_std.fillna(max_mag).median()), search_max_std=max_mag,
                no_flip_cases=int((~df.dnn_class_flipped).sum()), n_cases=int(len(df)),
                dnn_flip_rate=float(df.dnn_class_flipped.mean()), verdict_flip_rate=float(df.verdict_flipped.mean()))


def eval_drift(train, test):
    rows = []
    import joblib
    fe = joblib.load(C.PREPROCESSOR_PATH).named_steps["fe"]
    a, b = fe.transform(train[C.INPUT_COLUMNS]), fe.transform(test[C.INPUT_COLUMNS])
    for f in NUMERIC_FEATURES:
        v = psi(a[f].astype(float).values, b[f].astype(float).values)
        rows.append(dict(feature=f, psi=v, status="DRIFT" if v > 0.25 else ("WATCH" if v > 0.10 else "STABLE")))
    df = pd.DataFrame(rows); df.to_csv(R / "drift_baseline.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(df.feature, df.psi, color=BLUE); ax.tick_params(axis="y", labelsize=7); ax.axvline(0.25, c=RED, ls="--", label="Retrain trigger (0.25)")
    ax.axvline(0.10, c=ORANGE, ls=":", label="Watch (0.10)"); ax.invert_yaxis()
    ax.set_xlabel("PSI (train vs test)"); ax.set_title("Drift baseline"); ax.legend(); save_fig("drift_chart.png")
    return float(df.psi.max())


def write_readiness():
    m = json.loads((R / "metrics.json").read_text()); lat = json.loads((R / "latency.json").read_text())
    st = json.loads((R / "shap_stability.json").read_text()); rob = json.loads((R / "robustness_summary.json").read_text())
    cost = json.loads((R / "cost_summary.json").read_text()); fg = pd.read_csv(R / "fairness_amount.csv").iloc[-1]
    fa = pd.read_csv(R / "fairness_time.csv").iloc[-1]; dr = pd.read_csv(R / "drift_baseline.csv")
    tr = R / "testing_report.csv"
    if tr.exists():
        t = pd.read_csv(tr); n_pass = int((t.result == "PASSED").sum()); tests = (f"{n_pass}/{len(t)} passed", n_pass == len(t))
    else:
        tests = ("not run", False)
    dnn_cost = m["dnn"]["confusion_matrix"]["fn"] * C.COST_FN + m["dnn"]["confusion_matrix"]["fp"] * C.COST_FP
    meta_ok = all(k in json.loads(C.METADATA_PATH.read_text()) for k in ["feature_list", "threshold", "library_versions", "trained_at", "dataset_rows"])
    checks = [
        ("DNN predictive power", "test ROC-AUC >= 0.70", f"{m['dnn']['roc_auc']:.3f}", m["dnn"]["roc_auc"] >= 0.70),
        ("LR predictive power", "test ROC-AUC >= 0.70", f"{m['lr']['roc_auc']:.3f}", m["lr"]["roc_auc"] >= 0.70),
        ("DNN precision-recall (rare class)", "test PR-AUC >= 0.50", f"{m['dnn']['pr_auc']:.3f}", m["dnn"]["pr_auc"] >= 0.50),
        ("Flag rate sanity", "DNN flag rate <= 50%", f"{m['dnn']['flag_rate']:.1%}", m["dnn"]["flag_rate"] <= 0.5),
        ("False-positive rate", "DNN FPR <= 30%", f"{m['dnn']['confusion_matrix']['fp'] / max(m['dnn']['confusion_matrix']['fp'] + m['dnn']['confusion_matrix']['tn'], 1):.1%}",
         m["dnn"]["confusion_matrix"]["fp"] / max(m["dnn"]["confusion_matrix"]["fp"] + m["dnn"]["confusion_matrix"]["tn"], 1) <= 0.30),
        ("Expected cost vs flag-all baseline", "DNN cost < flag-everything cost", f"${dnn_cost:,.0f} vs ${cost['flag_all_cost']:,.0f}", dnn_cost < cost["flag_all_cost"]),
        ("Single-row latency (no SHAP)", "p95 <= 100 ms", f"{lat['single_no_explanation_ms']['p95']:.1f} ms", lat["single_no_explanation_ms"]["p95"] <= 100),
        ("Single-row latency (with SHAP)", "p95 <= 2000 ms", f"{lat['single_with_shap_ms']['p95']:.0f} ms", lat["single_with_shap_ms"]["p95"] <= 2000),
        ("Batch throughput", ">= 100 rows/s", f"{lat['batch']['rows_per_second']:.0f} rows/s", lat["batch"]["rows_per_second"] >= 100),
        ("Explanation stability", "mean Spearman >= 0.80", f"{st['mean_spearman']:.3f} +/- {st['std_spearman']:.3f}", st["mean_spearman"] >= 0.80),
        ("Subgroup recall (amount band)", "recall gap <= 0.20", f"{fg.tpr:.3f}", fg.tpr <= 0.20),
        ("Subgroup recall (time of day)", "recall gap <= 0.20", f"{fa.tpr:.3f}", fa.tpr <= 0.20),
        ("Robustness", "median flip distance >= 0.5 std",
         f"{'>= ' if rob['median_distance'] >= rob['search_max_std'] else ''}{rob['median_distance']:.3f} std ({rob['no_flip_cases']}/{rob['n_cases']} never flipped)",
         rob["median_distance"] >= 0.5),
        ("Data drift (train vs test)", "max PSI <= 0.25", f"{dr.psi.max():.3f}", dr.psi.max() <= 0.25),
        ("Automated tests", "all pass", tests[0], tests[1]),
        ("Model artifacts & metadata", "all files present, metadata complete", "complete" if meta_ok and all(p.exists() for p in [C.LR_PATH, C.DNN_PATH, C.PREPROCESSOR_PATH]) else "missing", meta_ok),
    ]
    df = pd.DataFrame([dict(check=a, criterion=b, measured=c, status="PASS" if d else "FAIL") for a, b, c, d in checks])
    df.to_csv(R / "deployment_readiness.csv", index=False)
    print(df.to_string(index=False))
    return df


def main():
    R.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
    pipe = FraudPipeline()
    train = pipe.train; test = pd.read_csv(C.TEST_CSV)
    y, dp, lp, X, trust = eval_metrics(pipe, test)
    print("metrics done"); (R / "cost_summary.json").write_text(json.dumps(eval_cost(y, dp), indent=2))
    print("cost done"); eval_fairness(test, y, dp, pipe.threshold); print("fairness done")
    print("max PSI", eval_drift(train, test))
    (R / "robustness_summary.json").write_text(json.dumps(eval_robustness(pipe, test), indent=2)); print("robustness done")
    eval_latency(pipe, test); print("latency done")
    eval_shap(pipe, test); print("shap done")
    write_readiness()


if __name__ == "__main__":
    if "--readiness" in sys.argv:
        write_readiness()
    else:
        main()
