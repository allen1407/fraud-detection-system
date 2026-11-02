"""Export models + reports to static JSON for the web frontend (web/), which scores entirely in the browser.

Run after scripts/run_all.sh:  .venv/bin/python scripts/export_web.py
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C                       # noqa: E402
from src.features import NUMERIC_FEATURES         # noqa: E402
from src.pipeline import FACTOR_LABELS, FraudPipeline  # noqa: E402

OUT = C.ROOT / "web" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)


def rnd(a, d=6):
    return np.round(np.asarray(a, dtype=float), d).tolist()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, separators=(",", ":")))


pipe = FraudPipeline(build_explainer=True)
scaler = pipe.ct.named_transformers_["num"]

# ---------------------------------------------------------------- model
layers = [dict(W=rnd(l.get_weights()[0], 7), b=rnd(l.get_weights()[1], 7), act=l.get_config()["activation"])
          for l in pipe.dnn.layers if l.__class__.__name__ == "Dense"]
dump("model.json", dict(
    features=NUMERIC_FEATURES, labels=FACTOR_LABELS, inputs=C.INPUT_COLUMNS,
    mean=rnd(scaler.mean_, 9), scale=rnd(scaler.scale_, 9),
    dnn=layers, lr=dict(coef=rnd(pipe.lr.coef_[0], 9), intercept=float(pipe.lr.intercept_[0])),
    threshold=pipe.threshold, lr_threshold=pipe.lr_threshold, trust_min=C.TRUST_MIN, knn_k=C.KNN_K,
    ref_distance=pipe.ref_distance, ranges=C.RANGES, recommendations=C.RECOMMENDATIONS,
    shap_background=rnd(pipe._bg, 6), trained_at=pipe.meta["trained_at"],
))
# kNN reference set (already standardised), float32 little-endian
Xref = pipe.pre.transform(pipe.reference[C.INPUT_COLUMNS]).astype("<f4")
(OUT / "reference.bin").write_bytes(Xref.tobytes())

# ---------------------------------------------------------------- parity fixtures + presets
test = pd.read_csv(C.TEST_CSV)
pred = pd.read_csv(C.REPORTS_DIR / "test_predictions.csv")
thr = pipe.threshold
legit, fraud = pred.y_true == 0, pred.y_true == 1


def pick(mask, key, asc):
    return int(pred[mask].sort_values(key, ascending=asc).index[0])


review = pred.verdict == "MANUAL REVIEW"
idx = {"Clear Legit": pick(legit & (pred.verdict == "AUTO-APPROVE"), "dnn_prob", True),
       "Clear Fraud": pick(fraud & (pred.verdict == "AUTO-BLOCK"), "dnn_prob", False),
       "Borderline": int(pred[review].assign(d=lambda x: (x.dnn_prob - thr).abs()).sort_values("d").index[0])}
presets = {k: {c: float(test.loc[i, c]) for c in C.INPUT_COLUMNS} for k, i in idx.items()}

rng = np.random.default_rng(C.SEED)
sample_idx = list(idx.values()) + rng.choice(pred[pred.verdict == "AUTO-BLOCK"].index, 2, replace=False).tolist() \
    + rng.choice(pred[review].index, 3, replace=False).tolist() + rng.choice(pred[legit].index, 3, replace=False).tolist()
recent = []
for i in dict.fromkeys(sample_idx):
    rec = {c: float(test.loc[i, c]) for c in C.INPUT_COLUMNS}
    r = pipe.assess(rec, explain=False)
    recent.append(dict(id=f"TX-{100000 + int(i)}", record=rec, label=int(test.loc[i, C.TARGET]),
                       dnn_prob=r["dnn_prob"], lr_prob=r["lr_prob"], trust=r["trust"], verdict=r["verdict"]))
dump("samples.json", dict(presets=presets, recent=recent))

# ---------------------------------------------------------------- reports
m = json.loads((C.REPORTS_DIR / "metrics.json").read_text())
y = pred.y_true.values
curves = {}
for k in ("dnn", "lr"):
    f, t, _ = roc_curve(y, pred[f"{k}_prob"])
    p, r, _ = precision_recall_curve(y, pred[f"{k}_prob"])
    # keep only the corners of the step curves (recall/TPR only changes at the 71 frauds)
    corners = lambda a: [i for i in range(len(a)) if i in (0, len(a) - 1) or a[i] != a[i - 1] or a[i] != a[i + 1]]  # noqa: E731
    curves[k] = dict(roc=[[float(f[i]), float(t[i])] for i in corners(t)],
                     pr=[[float(r[i]), float(p[i])] for i in corners(r)])
sweep = []
for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
    flag = pred.dnn_prob.values >= t
    sweep.append(dict(t=float(t), tp=int((flag & (y == 1)).sum()), fp=int((flag & (y == 0)).sum()),
                      fn=int((~flag & (y == 1)).sum()), tn=int((~flag & (y == 0)).sum())))
hours = (test[C.TIME_COL] // 3600).astype(int).clip(0, 47)
traffic = [dict(h=int(h), n=int((hours == h).sum()), fraud=int(((hours == h) & (test[C.TARGET] == 1)).sum()))
           for h in range(48)]


def csv(name):
    return json.loads(pd.read_csv(C.REPORTS_DIR / name).to_json(orient="records"))


dump("reports.json", dict(
    metrics=m, curves=curves, sweep=sweep, traffic=traffic,
    cost=json.loads((C.REPORTS_DIR / "cost_summary.json").read_text()), cost_fn=C.COST_FN, cost_fp=C.COST_FP,
    latency=json.loads((C.REPORTS_DIR / "latency.json").read_text()),
    data=json.loads((C.PROCESSED_DIR / "data_summary.json").read_text()), n_features=len(pipe.meta["feature_list"]),
    fairness_amount=csv("fairness_amount.csv"), fairness_time=csv("fairness_time.csv"),
    robustness=csv("robustness.csv"), robustness_summary=json.loads((C.REPORTS_DIR / "robustness_summary.json").read_text()),
    shap_stability=json.loads((C.REPORTS_DIR / "shap_stability.json").read_text()), shap_global=csv("shap_global.csv"),
    drift=csv("drift_baseline.csv"), readiness=csv("deployment_readiness.csv"), testing=csv("testing_report.csv"),
))

# ---------------------------------------------------------------- docs + sample batch
docs = C.ROOT / "web" / "public" / "docs"
shutil.rmtree(docs, ignore_errors=True)
shutil.copytree(C.DOCS_DIR, docs)
shutil.copytree(C.FIGURES_DIR, docs / "figures")
for p in docs.glob("*.md"):
    p.write_text(p.read_text().replace("../reports/figures/", "figures/"))
dump("docs.json", sorted(p.name for p in docs.glob("*.md")))
shutil.copy(C.SAMPLES_DIR / "sample_batch.csv", OUT / "sample_batch.csv")
print(f"Exported web data to {OUT}")
