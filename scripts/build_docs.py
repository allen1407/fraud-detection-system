"""Generate docs/*.md and the README results from reports/ so every number comes from the real run."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, D = ROOT / "reports", ROOT / "docs"
J = lambda n: json.loads((R / n).read_text())
m, lat, st, rob, cost = J("metrics.json"), J("latency.json"), J("shap_stability.json"), J("robustness_summary.json"), J("cost_summary.json")
meta = json.loads((ROOT / "models" / "metadata.json").read_text())
data = json.loads((ROOT / "data" / "processed" / "data_summary.json").read_text())
fg, fa = pd.read_csv(R / "fairness_amount.csv"), pd.read_csv(R / "fairness_time.csv")
pred = pd.read_csv(R / "test_predictions.csv")
drift, ready, tests = pd.read_csv(R / "drift_baseline.csv"), pd.read_csv(R / "deployment_readiness.csv"), pd.read_csv(R / "testing_report.csv")
rb, shg = pd.read_csv(R / "robustness.csv"), pd.read_csv(R / "shap_global.csv")
d, l = m["dnn"], m["lr"]
cmd, cml = d["confusion_matrix"], l["confusion_matrix"]
n_fail = int((ready.status == "FAIL").sum())
fail_names = ", ".join(ready[ready.status == "FAIL"].check)
n_fraud_test = cmd["fn"] + cmd["tp"]
n_legit_test = cmd["tn"] + cmd["fp"]
rt = m["routing"]; n_review = rt.get("MANUAL REVIEW", 0)
review_low_trust = int(((pred.verdict == "MANUAL REVIEW") & (pred.trust < 1.0)).sum())
dflag, lflag = pred.dnn_prob >= d["threshold"], pred.lr_prob >= l["threshold"]
n_disagree = int((dflag != lflag).sum())
rf = (pred.y_true == 1) & (pred.verdict == "MANUAL REVIEW")
rf_both, rf_one, rf_none = int((rf & dflag & lflag).sum()), int((rf & (dflag != lflag)).sum()), int((rf & ~dflag & ~lflag).sum())
fraud_routes = pred[pred.y_true == 1].verdict.value_counts().to_dict()
dc = m["dnn_at_cost_optimal"]["confusion_matrix"]
rob_fraud = rb[rb.is_fraud == 1]; rob_missed = int((rob_fraud.original_prob < d["threshold"]).sum())
seg_fraud = pd.concat([fg.iloc[:-1].n_fraud, fa.iloc[:-1].n_fraud])
md = lambda df: df.to_markdown(index=False) if hasattr(df, "to_markdown") else str(df)
try:
    import tabulate  # noqa
except ImportError:
    def md(df):  # minimal markdown table without tabulate
        cols = list(df.columns)
        rows = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
        for _, r in df.iterrows():
            rows.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
        return "\n".join(rows)

HEADLINE = f"""| Metric | DNN (threshold {d['threshold']:.2f}) | Logistic regression (0.50) |
|---|---|---|
| ROC-AUC | {d['roc_auc']:.3f} | {l['roc_auc']:.3f} |
| PR-AUC | {d['pr_auc']:.3f} | {l['pr_auc']:.3f} |
| F1 | {d['f1']:.3f} | {l['f1']:.3f} |
| Precision | {d['precision']:.3f} | {l['precision']:.3f} |
| Recall | {d['recall']:.3f} | {l['recall']:.3f} |
| Accuracy | {d['accuracy']:.3f} | {l['accuracy']:.3f} |
| Flag rate | {d['flag_rate']:.2%} | {l['flag_rate']:.2%} |
| Confusion matrix (TN/FP/FN/TP) | {cmd['tn']}/{cmd['fp']}/{cmd['fn']}/{cmd['tp']} | {cml['tn']}/{cml['fp']}/{cml['fn']}/{cml['tp']} |

Test set: {m['n_test']:,} transactions ({m['test_fraud_rate']:.2%} fraud, {n_fraud_test} frauds). Source: `reports/metrics.json`."""

HONESTY = f"""> **Key finding.** Both models detect fraud well in ranking terms: test ROC-AUC {d['roc_auc']:.3f} (DNN) and {l['roc_auc']:.3f} (LR), PR-AUC {d['pr_auc']:.3f} and {l['pr_auc']:.3f} against a fraud base rate of only {m['test_fraud_rate']:.2%}. At its operating threshold the DNN catches {cmd['tp']} of {n_fraud_test} test frauds with {cmd['fp']} false alarms among {n_legit_test:,} legitimate transactions. The simple logistic regression ranks slightly better than the DNN; the DNN's advantage is higher recall at the chosen threshold. Caveats: only {n_fraud_test} frauds in the test set (wide uncertainty), anonymised PCA features (explanations are not business-readable), two days of 2013 data, a random rather than temporal split, and no customer attributes for a fairness audit."""

THRESH = f"""The operating DNN threshold is **{meta['threshold']:.2f}**: the F1-maximising threshold on the validation split, subject to a validation false-positive-rate guardrail of {meta['max_val_fpr_guardrail']:.0%} (the guardrail is not binding here; the unconstrained F1 optimum is also {meta['threshold_f1_unconstrained']:.2f}). With a missed fraud costing ${meta['cost_fn']:.0f} and a false alarm ${meta['cost_fp']:.0f}, the cost-optimal validation threshold is lower, **{meta['threshold_cost_optimal']:.2f}**. On the test set it catches {dc['tp']} frauds with {dc['fp']} false alarms (cost ${dc['fn'] * meta['cost_fn'] + dc['fp'] * meta['cost_fp']:,.0f}), against {cmd['tp']} frauds and {cmd['fp']} false alarms (cost ${cmd['fn'] * meta['cost_fn'] + cmd['fp'] * meta['cost_fp']:,.0f}) at {meta['threshold']:.2f}. Baselines: approving everything costs ${cost['approve_all_cost']:,.0f}; flagging everything ${cost['flag_all_cost']:,.0f}. The F1 threshold is kept as the operating point because it keeps the alert volume lowest; a bank that prices misses at $500 should consider switching to {meta['threshold_cost_optimal']:.2f}. All thresholds are recorded in `models/metadata.json`."""

DATASET = f"""Kaggle **Credit Card Fraud Detection** dataset (ULB Machine Learning Group): transactions made by European cardholders over two days in September 2013. Raw file `creditcard.csv`: {data['raw_rows']:,} rows; {data['duplicates_removed']:,} exact duplicates removed, leaving **{data['rows']:,} rows** with **{data['fraud_rows']} frauds ({data['fraud_rate']:.3%})**, no missing values. Columns: `Time` (seconds since the first transaction, {data['time_span_hours']:.0f} h span), `V1`-`V28` (PCA components; the original features are confidential), `Amount` (max ${data['amount_max']:,.2f}) and the label `Class` (1 = fraud). Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud (licence: Open Database License, ODbL)."""

def w(name, text):
    (D / name).write_text(text.strip() + "\n"); print("wrote", name)


w("SRS.md", f"""
# Software Requirements Specification — Intelligent Fraud Detection System

## 1. Purpose and scope
A decision-support system that validates a credit-card transaction, scores it with two models, explains the result and recommends AUTO-BLOCK, AUTO-APPROVE or MANUAL REVIEW. Scope: Task 9 (integration & deployment) and Task 10 (evaluation, responsible AI, documentation). Training data: {data['rows']:,} credit-card transactions ({data['fraud_rate']:.3%} fraud) over {data['time_span_hours']:.0f} hours.

## 2. Functional requirements
| ID | Requirement | Implemented in | Verified by |
|---|---|---|---|
| FR1 | Validate input: all 30 fields (Time, V1-V28, Amount) present, numeric and finite; no negative amount or time; Time within the 48 h window; V1-V28 within ±150; return a list of errors | `FraudPipeline.validate` | `test_negative_amount_rejected`, `test_missing_field_rejected`, `test_out_of_range_and_non_numeric_rejected` |
| FR2 | Score with the DNN (saved threshold) and LR (0.5) | `FraudPipeline.score` | `test_probabilities_in_unit_interval` |
| FR3 | Explain with the top-5 SHAP factors, direction and signed value | `FraudPipeline.explain` | `test_shap_top5_has_direction_and_sign` |
| FR4 | Compute a trust score from nearest-neighbour distance to the training reference set | `FraudPipeline.trust_score` | `test_trust_score_positive` |
| FR5 | Route: AUTO-BLOCK / AUTO-APPROVE only when both models agree and trust ≥ 1.0, otherwise MANUAL REVIEW | `FraudPipeline.route` | `test_routing_rules` |
| FR6 | Generate a text report (and PDF in the app); invalid input produces "TRANSACTION REJECTED" with reasons | `generate_report` | `test_report_nonempty_for_valid_and_invalid` |
| FR7 | Batch scoring that flags invalid rows and never crashes | `score_batch` | `test_batch_of_20_runs_with_invalid_rows`, `test_batch_vs_single_parity` |
| FR8 | Web UI: single scoring, batch, performance, responsible AI, monitoring, documentation | `app.py`, `pages/` | manual walkthrough + screenshots |

## 3. Non-functional requirements
| ID | Requirement | Criterion | Measured |
|---|---|---|---|
| NFR1 | Single-row latency (no SHAP) | p95 ≤ 100 ms | {lat['single_no_explanation_ms']['p95']:.1f} ms |
| NFR2 | Single-row latency (with SHAP) | p95 ≤ 2000 ms | {lat['single_with_shap_ms']['p95']:.0f} ms |
| NFR3 | Batch throughput | ≥ 100 rows/s | {lat['batch']['rows_per_second']:,.0f} rows/s |
| NFR4 | Explanation stability under 1% noise | mean Spearman ≥ 0.80 | {st['mean_spearman']:.3f} |
| NFR5 | Reproducibility | fixed seed 42, pinned `requirements.txt`, `scripts/run_all.sh` | yes |
| NFR6 | Usability | friendly errors, no stack traces in the UI | yes |
| NFR7 | Privacy | no transactions persisted by the app; reports are generated on demand; input features are already anonymised (PCA) | yes |
| NFR8 | Portability | local, Docker, Hugging Face Spaces | see DEPLOYMENT_GUIDE |

## 4. Constraints and assumptions
Cost assumptions: false negative ${meta['cost_fn']:.0f}, false positive ${meta['cost_fp']:.0f}. Python 3.11, TensorFlow 2.16, scikit-learn 1.5.

{HONESTY}
""")

w("SDD.md", f"""
# Software Design Description

## 1. Architecture
```mermaid
flowchart LR
    A[Input: form / CSV] --> B[validate]
    B -- invalid --> X[TRANSACTION REJECTED + reasons]
    B -- valid --> C[FeatureBuilder + ColumnTransformer]
    C --> D[Keras DNN]
    C --> E[Logistic regression]
    C --> F[kNN trust score]
    D --> G[SHAP top-5]
    D --> H{{route}}
    E --> H
    F --> H
    H --> I[AUTO-BLOCK / AUTO-APPROVE / MANUAL REVIEW]
    I --> J[report .txt / .pdf]
```

## 2. Modules
| Module | Responsibility |
|---|---|
| `src/config.py` | paths, seed 42, costs, schema (Time, V1-V28, Amount), ranges, trust reference size, verdict strings |
| `src/data.py` | load/clean (drop duplicates), stratified 70/15/15 split ({meta['split_sizes']['train']}/{meta['split_sizes']['val']}/{meta['split_sizes']['test']}) |
| `src/features.py` | `FeatureBuilder` (hour of day and a night flag derived from `Time`, `log_amount`; raw `Time` is dropped because "seconds since the dataset started" has no meaning for new transactions) + `ColumnTransformer` (StandardScaler), fit on train only |
| `src/train.py` | LR + DNN (64-32-16, dropout 0.3, L2, batch 512, early stopping on validation loss; class imbalance handled by threshold choice rather than class weights), threshold selection, sanity gate |
| `src/pipeline.py` | `FraudPipeline`: validate, score, explain, trust_score, route, generate_report, score_batch |
| `src/evaluate.py` | all metrics, figures, subgroup analysis, robustness, drift, latency, SHAP stability, readiness |
| `src/ui.py`, `app.py`, `pages/` | Streamlit UI (cached resources, presets, charts); calls only `src/` |
| `scripts/` | `prepare_data.py`, `run_all.sh`, `build_docs.py`, `build_report_pdf.py`, `screenshots.py` |

## 3. Inference data flow
1. `validate(record)` returns a list of errors (empty when valid).
2. `FeatureBuilder` derives the {len(meta['feature_list'])}-column model input (after one-hot encoding) with the saved preprocessor.
3. DNN and LR produce probabilities; the DNN is compared with threshold {meta['threshold']:.2f}, LR with 0.50.
4. Trust = (95th percentile of leave-one-out mean 5-NN distances in the reference set) ÷ (mean 5-NN distance of the query), capped at 10. The reference set holds every training fraud plus a fixed sample of 20,000 legitimate training rows (keeps the index small enough to load in seconds). Values ≥ 1.0 mean the transaction is as familiar as 95% of reference rows.
5. `route` combines agreement and trust. Explanation uses `shap.KernelExplainer` on the {len(meta['feature_list'])} engineered features (V1-V28, Amount, hour, night flag, log amount) with a 100-row background sample built once.

## 4. Thresholds
{THRESH}

## 5. Persistence
| Artifact | File |
|---|---|
| Preprocessor (FeatureBuilder + ColumnTransformer) | `models/preprocessor.pkl` |
| Logistic regression | `models/fraud_lr_model.pkl` |
| DNN | `models/fraud_dnn_model.keras` |
| Feature list, thresholds, library versions, training date ({meta['trained_at'][:10]}), dataset rows ({meta['dataset_rows']:,}) | `models/metadata.json` |
| Splits and cleaned data | `data/processed/` |
| Evaluation outputs | `reports/` |

The model and preprocessor must always be deployed as a matched pair.
""")

w("USER_MANUAL.md", """
# User Manual

Start the app: `streamlit run app.py` (see DEPLOYMENT_GUIDE). Use the sidebar to switch pages.

## Home
Project overview, dataset summary, architecture and headline test metrics.

![Home](screenshots/01_home.png)

## Scoring a single transaction
1. Open **Score Transaction**.
2. Pick an example in **Load example** (a real test-set transaction: Clear Legit, Clear Fraud or Borderline), or type your own values: `Time` (seconds since the start of the 48-hour window), `Amount`, and the 28 PCA components in the *V1-V28* section.
3. Click **Predict**.
4. Read the coloured verdict (red AUTO-BLOCK, green AUTO-APPROVE, amber MANUAL REVIEW, grey TRANSACTION REJECTED), the DNN and LR fraud probabilities, the trust score, the recommendation and the top-5 SHAP factors (red raises fraud risk, green lowers it). Open **Full report** for the complete text report.
5. If the input is invalid (for example a negative amount) the transaction is rejected and the reasons are listed.

![Prediction](screenshots/02_score_prediction.png)
![Rejected input](screenshots/03_score_rejected.png)

## Batch scoring
1. Open **Batch Scoring** and download the CSV template (or the 20-row sample, which includes two deliberately invalid rows).
2. Upload a CSV containing every required column.
3. Optionally tick *Include SHAP top-5 factors* (slower).
4. Watch the progress bar, inspect the results table and verdict chart, then download the results CSV. Invalid rows show their reasons in the `errors` column.

![Batch](screenshots/04_batch.png)

## Other pages
**Model Performance** (metrics, ROC/PR, interactive threshold slider, latency), **Responsible AI** (fairness, robustness, SHAP), **Monitoring & Readiness** (PSI drift, readiness checklist, retraining/rollback), **Documentation** (these documents).

![Performance](screenshots/05_performance.png)
![Responsible AI](screenshots/06_responsible_ai.png)
![Monitoring](screenshots/07_monitoring.png)

## Troubleshooting
"Models or reports are missing": run `bash scripts/run_all.sh`. "File is missing required columns": use the CSV template.
""")

w("RESPONSIBLE_AI_REPORT.md", f"""
# Responsible AI Report

{HONESTY}

## 1. Fairness and subgroup performance
The dataset contains **no customer attributes** (no gender, age, nationality or location), so fairness across protected groups cannot be measured, and the model cannot use protected attributes directly. Because V1-V28 are PCA mixtures of undisclosed original features, proxies for protected attributes may still be embedded in them; this cannot be checked without the original data. As a substitute, performance is compared across transaction segments for the DNN at threshold {d['threshold']:.2f} on the test set (`reports/fairness_amount.csv`, `reports/fairness_time.csv`).

**By amount band**

{md(fg.round(4))}

**By time of day**

{md(fa.round(4))}

Largest recall (TPR) gaps: amount band {fg.iloc[-1].tpr:.3f}, time of day {fa.iloc[-1].tpr:.3f} (criterion 0.20). Each segment holds only {seg_fraud.min()}-{seg_fraud.max()} frauds, so one or two transactions move recall by 0.1 or more; the gaps are not statistically conclusive but are flagged for re-checking on more data. False-positive rates are below 0.2% in every segment.

## 2. Robustness
Counterfactual search over V1-V28 and Amount for 15 test cases (5 fraud, 10 legitimate), 400 random directions per magnitude, magnitudes up to {rob['search_max_std']:.0f} training standard deviations (`reports/robustness.csv`). The DNN class flipped in {rob['dnn_flip_rate']:.0%} of cases and the routed verdict in {rob['verdict_flip_rate']:.0%}; {rob['no_flip_cases']} of {rob['n_cases']} cases never flipped inside the search radius (median distance reported as ≥ {rob['median_distance']:.1f} SD). This shows the decisions are not sensitive to random noise. It does **not** show adversarial robustness: random directions in 29 dimensions rarely point at the decision boundary, and a gradient-based attacker would need far less. Note also that {rob_missed} of the 5 sampled frauds were already missed by the model (probability below the threshold); random perturbation leaves them missed.

## 3. Explainability
SHAP (KernelExplainer, 100-row background) gives the top-5 factors per decision. Stability: mean Spearman rank correlation {st['mean_spearman']:.3f} ± {st['std_spearman']:.3f} (min {st['min_spearman']:.3f}) under 1% noise on V1-V28 and Amount over {st['n_samples']} samples (`reports/shap_stability.json`). Global importance (mean |SHAP|, top 5): {', '.join(f"{r.feature} ({r.mean_abs_shap:.2e})" for r in shg.head(5).itertuples())}. Mean |SHAP| values are tiny because almost every transaction scores near 0; for flagged transactions the attributions are orders of magnitude larger. The top SHAP components (V14, V12, V10) are also among the strongest class separators in the EDA notebook, so the explanations are consistent with the data. They are, however, **not business-interpretable**: "V14 is low" cannot be explained to a customer or mapped to a rule without the confidential PCA loadings.

## 4. Privacy
The app stores no transactions: inputs live only in the session, reports are generated on demand, and nothing is sent to external services. The dataset was anonymised by the provider (PCA transformation, no identifiers, no customer attributes), so it carries little re-identification risk. A production system fed with raw card data would need access control, encryption and PCI-DSS compliance. The raw and processed CSVs are excluded from git (`.gitignore`) because of their size.

## 5. Transparency
Every decision comes with both probabilities, model agreement, trust score, SHAP factors and a plain-language recommendation. `models/metadata.json` records features, thresholds, library versions, training date and dataset size. Known weaknesses are shown in the app (Monitoring page and Documentation).

## 6. Human oversight
The system never acts alone on uncertain cases. Automatic action requires both models to agree **and** trust ≥ 1.0. On the test set, {n_review:,} of {m['n_test']:,} transactions ({n_review / m['n_test']:.1%}) were routed to MANUAL REVIEW, {rt.get('AUTO-APPROVE', 0):,} auto-approved and {rt.get('AUTO-BLOCK', 0)} auto-blocked. Low trust drives the queue: {review_low_trust:,} review cases have trust < 1.0 (unusual transactions), while the models disagree on only {n_disagree} test transactions. Of the {n_fraud_test} test frauds, {fraud_routes.get('AUTO-BLOCK', 0)} were auto-blocked, {fraud_routes.get('MANUAL REVIEW', 0)} reached an analyst and {fraud_routes.get('AUTO-APPROVE', 0)} were auto-approved (missed). Of the frauds sent to review, {rf_both} were flagged by both models but held back because they look unlike the reference data (frauds are unusual by nature), {rf_one} were flagged by one model only, and {rf_none} were flagged by neither and reached an analyst only because of low trust. The price is an analyst queue of about {n_review / m['n_test']:.1%} of traffic, almost all legitimate; `TRUST_MIN` in `src/config.py` controls this trade-off.

## 7. Threshold choice
{THRESH}

## 8. Limitations
- Only {data['fraud_rows']} frauds in total ({n_fraud_test} in the test set): recall and precision carry wide confidence intervals (one fraud moves test recall by {1 / n_fraud_test:.1%}).
- Features are anonymised PCA components: explanations cannot be given to customers in plain language, and proxies for protected attributes cannot be ruled out.
- Two days of 2013 European card traffic; fraud patterns change quickly, so performance on current traffic is unknown.
- Random (not temporal) split: train and test overlap in time, so metrics and the drift baseline are optimistic.
- No customer attributes: fairness is approximated with transaction segments only.
- SHAP explains the DNN only and is an approximation.
""")

dep = ready.copy()
w("DEPLOYMENT_GUIDE.md", f"""
# Deployment Guide

## Local
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_all.sh          # prepare data, train, evaluate, test, notebook, report
streamlit run app.py
```
On macOS `requirements.txt` installs `tensorflow`; on Linux it installs `tensorflow-cpu`.

## Docker
```bash
docker build -t fraud-detection .
docker run -p 8501:8501 fraud-detection        # http://localhost:8501
```
The image copies the working directory, so run `bash scripts/run_all.sh` first: the trained models, reports and the processed CSVs (not in git) must exist locally. Rebuild after retraining.

## Hugging Face Spaces
1. Create a Space with SDK **Streamlit** (or **Docker** and reuse the Dockerfile with `app_port: 8501`).
2. Push the repo including `models/` and `reports/`. The app also needs `data/processed/train.csv` (~105 MB, trust reference and SHAP background) and `data/processed/test.csv` (~23 MB, examples) at runtime; they are git-ignored, so either track them with Git LFS or run `python scripts/prepare_data.py data/raw/creditcard.csv && python -m src.train` as a build step.
3. Set the Python version to 3.11 and keep `requirements.txt` pinned.
Do not host real customer data on a public Space.

## Monitoring
- **Monthly:** compute PSI of each model feature against the training distribution (`src/evaluate.py` shows the method; baseline in `reports/drift_baseline.csv`, max PSI {drift.psi.max():.3f} = {drift.sort_values('psi').iloc[-1].feature}).
- **Retrain trigger:** any PSI > 0.25.
- **Schedule:** retrain quarterly regardless.
- Track the flag rate, the MANUAL REVIEW share and analyst-confirmed fraud rate.

## Rollback
Keep the previous `fraud_lr_model.pkl`, `fraud_dnn_model.keras`, `preprocessor.pkl` and `metadata.json` together in a dated folder (e.g. `models/archive/YYYY-MM-DD/`). To roll back, restore the whole set at once; never mix a model with a different preprocessor.

## Readiness status
{md(dep)}

**{n_fail} of {len(ready)} checks fail** ({fail_names}). The predictive, cost, latency and engineering checks pass; the subgroup recall gaps rest on a handful of frauds per segment and must be re-checked (or formally accepted) before production. Independently of the checklist, the model must be re-validated on recent traffic before any real use.
""")

w("TESTING_REPORT.md", f"""
# Testing Report

Run: `python -m pytest -q`; results exported to `reports/testing_report.csv`. {int((tests.result == 'PASSED').sum())}/{len(tests)} tests passed.

{md(tests)}

Latency criterion: p95 single-row latency (no SHAP) < 100 ms; measured by `src/evaluate.py` at {lat['single_no_explanation_ms']['p95']:.1f} ms (`reports/latency.json`).
""")

w("FINAL_REPORT.md", f"""
# Task 9 & 10 Report — Intelligent Fraud Detection System

## 1. System integration
`FraudPipeline` (`src/pipeline.py`) integrates validation, feature engineering, two models (Keras DNN, logistic regression), SHAP explanations, a nearest-neighbour trust score, human-in-the-loop routing and report generation. A Streamlit app (`app.py`, `pages/`) exposes it through six pages and only calls `src/`. Dataset: {data['rows']:,} credit-card transactions, {data['fraud_rows']} frauds ({data['fraud_rate']:.3%}); split 70/15/15 stratified ({meta['split_sizes']['train']}/{meta['split_sizes']['val']}/{meta['split_sizes']['test']}); preprocessor fit on train only.

{HONESTY}

## 2. Deliverables inventory
| Deliverable | Path |
|---|---|
| Pipeline | `src/pipeline.py` |
| Features / data / training / evaluation | `src/features.py`, `src/data.py`, `src/train.py`, `src/evaluate.py` |
| App | `app.py`, `pages/`, `src/ui.py`, `.streamlit/config.toml` |
| Models | `models/fraud_lr_model.pkl`, `models/fraud_dnn_model.keras`, `models/preprocessor.pkl`, `models/metadata.json` |
| Metrics and CSVs | `reports/metrics.json`, `cost_threshold.csv`, `latency.json`, `shap_stability.json`, `fairness_amount.csv`, `fairness_time.csv`, `robustness.csv`, `drift_baseline.csv`, `deployment_readiness.csv`, `testing_report.csv` |
| Figures | `reports/figures/` |
| Tests | `tests/` |
| Documentation | `docs/SRS.md`, `SDD.md`, `USER_MANUAL.md`, `RESPONSIBLE_AI_REPORT.md`, `DEPLOYMENT_GUIDE.md`, `TESTING_REPORT.md`, `FINAL_REPORT.md` |
| Notebook | `notebooks/01_eda_and_training.ipynb` |
| Deployment | `Dockerfile`, `scripts/run_all.sh` |

## 3. Functional testing
{int((tests.result == 'PASSED').sum())}/{len(tests)} automated tests pass (valid score, negative amount rejected, missing field rejected, probabilities in [0,1], recommendation in allowed set, non-empty report, batch of 20, batch-vs-single parity, latency). Details: `docs/TESTING_REPORT.md`.

## 4. Performance evaluation
{HEADLINE}

![ROC](../reports/figures/roc_curve.png)
![Precision-recall](../reports/figures/pr_curve.png)
![Confusion matrices](../reports/figures/confusion_matrix.png)

{THRESH}

![Cost vs threshold](../reports/figures/cost_threshold_analysis.png)

Latency: single row {lat['single_no_explanation_ms']['mean']:.1f} ms mean / {lat['single_no_explanation_ms']['p95']:.1f} ms p95 without SHAP; {lat['single_with_shap_ms']['mean']:.0f} ms mean / {lat['single_with_shap_ms']['p95']:.0f} ms p95 with SHAP; batch {lat['batch']['rows_per_second']:,.0f} rows/s.

## 5. Explainability
Top-5 SHAP factors per decision; stability {st['mean_spearman']:.3f} ± {st['std_spearman']:.3f} Spearman under 1% noise. Most important features: {', '.join(shg.head(5).feature)}.

![SHAP](../reports/figures/shap_summary.png)

## 6. Responsible AI
No customer attributes exist, so subgroup performance is measured by amount band and time of day: recall gaps {fg.iloc[-1].tpr:.3f} and {fa.iloc[-1].tpr:.3f} (criterion 0.20; not met, based on {seg_fraud.min()}-{seg_fraud.max()} frauds per segment). Robustness: no DNN class flips within {rob['search_max_std']:.0f} SD of random perturbation ({rob['no_flip_cases']}/{rob['n_cases']} cases). Human review: {n_review:,} of {m['n_test']:,} test transactions ({n_review / m['n_test']:.1%}) go to MANUAL REVIEW; of the {n_fraud_test} test frauds {fraud_routes.get('AUTO-BLOCK', 0)} are auto-blocked, {fraud_routes.get('MANUAL REVIEW', 0)} reviewed and {fraud_routes.get('AUTO-APPROVE', 0)} missed. Full discussion in `docs/RESPONSIBLE_AI_REPORT.md`.

## 7. Deployment readiness
{md(ready)}

**{n_fail} of {len(ready)} checks fail** ({fail_names}). Predictive power, cost, latency, throughput, stability, robustness, drift, tests and artifacts pass.

## 8. Limitations
Few positives ({n_fraud_test} test frauds); anonymised PCA features (explanations not business-readable); two days of 2013 data and a random split (optimistic metrics, no real drift estimate); no customer attributes for fairness; SHAP explains the DNN only.

## 9. Future enhancements
Temporal validation (train on day 1, test on day 2) and re-validation on recent traffic; probability calibration; amount-weighted cost (a missed $2,000 fraud costs more than a missed $2 one); gradient-boosting comparison; richer behavioural features (velocity, device, merchant) if non-anonymised data is available; a fairness audit with customer attributes; scheduled PSI monitoring job and an analyst feedback loop; authentication and audit logging.
""")

# README results block

# ---------------------------------------------------------------- README
w_readme = f"""
# Intelligent Fraud Detection System

A Streamlit decision-support system that scores credit-card transactions with a Keras DNN and logistic regression, explains them with SHAP, and routes uncertain cases to human review (Computational Intelligence Lab, Tasks 9 and 10).

{HONESTY}

## Dataset
{DATASET}

Place `creditcard.csv` in `data/raw/` (it is git-ignored because of its 150 MB size).

## Architecture
```mermaid
flowchart LR
    A[Input: form / CSV] --> B[Validate + engineer features]
    B --> C[DNN + LR scoring]
    B --> D[SHAP top-5 + trust score]
    C --> E{{Route}}
    D --> E
    E --> F[AUTO-BLOCK / AUTO-APPROVE / MANUAL REVIEW]
    F --> G[Report .txt / .pdf]
```

## Task 9 deliverables
| Deliverable | File path | Status |
|---|---|---|
| Integrated pipeline (`FraudPipeline`) | `src/pipeline.py` | Done |
| Streamlit app (6 pages + home) | `app.py`, `pages/` | Done |
| Models + preprocessor + metadata | `models/` | Done |
| Batch sample / template | `data/samples/sample_batch.csv` | Done |
| Dockerfile | `Dockerfile` | Done (not built in this environment) |
| One-command run | `scripts/run_all.sh` | Done |
| Deployment guide | `docs/DEPLOYMENT_GUIDE.md` | Done |

## Task 10 deliverables
| Deliverable | File path | Status |
|---|---|---|
| Metrics, thresholds | `reports/metrics.json` | Done |
| Figures (ROC, PR, confusion, cost, SHAP, subgroups, drift) | `reports/figures/` | Done |
| Cost sweep | `reports/cost_threshold.csv` | Done |
| Latency | `reports/latency.json` | Done |
| SHAP stability | `reports/shap_stability.json` | Done |
| Subgroup / fairness analysis | `reports/fairness_amount.csv`, `reports/fairness_time.csv` | Done (no customer attributes in the data) |
| Robustness | `reports/robustness.csv` | Done |
| Drift baseline | `reports/drift_baseline.csv` | Done |
| Readiness checklist | `reports/deployment_readiness.csv` | Done ({n_fail} checks FAIL) |
| Tests + report | `tests/`, `reports/testing_report.csv` | Done ({int((tests.result == 'PASSED').sum())}/{len(tests)} pass) |
| SRS, SDD, user manual, RAI report, testing report, final report | `docs/` | Done |
| Final report PDF | `reports/Task9_10_Report.pdf` | Done |
| Notebook (executed) | `notebooks/01_eda_and_training.ipynb` | Done |

## Headline results (test set)
{HEADLINE}

Operating threshold: {meta['threshold']:.2f}, see the threshold note in `docs/SDD.md`. Readiness: {n_fail} of {len(ready)} checks FAIL ({fail_names}); see `docs/DEPLOYMENT_GUIDE.md`.

## Screenshots
![Home](docs/screenshots/01_home.png)
![Score transaction](docs/screenshots/02_score_prediction.png)
![Batch](docs/screenshots/04_batch.png)
![Performance](docs/screenshots/05_performance.png)
![Responsible AI](docs/screenshots/06_responsible_ai.png)
![Monitoring](docs/screenshots/07_monitoring.png)

## Quick start
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# 1. place creditcard.csv in data/raw/
python scripts/prepare_data.py data/raw/creditcard.csv   # 2. prepare data
python -m src.train                                   # 3. train
python -m src.evaluate                                # 4. evaluate
python -m pytest -q                                   #    tests
streamlit run app.py                                  # 5. run the app
```

### Everything in one command
```bash
bash scripts/run_all.sh
```

## Repository structure
```
fraud-detection-system/
├── README.md  requirements.txt  Dockerfile  pytest.ini  .gitignore
├── .streamlit/config.toml
├── app.py                      Home page
├── pages/                      Score, Batch, Performance, Responsible AI, Monitoring, Documentation
├── src/                        config, data, features, train, pipeline, evaluate, ui
├── scripts/                    prepare_data.py, run_all.sh, build_docs.py, build_report_pdf.py, screenshots.py
├── models/                     LR, DNN, preprocessor, metadata.json
├── data/                       raw/creditcard.csv, processed/ (both git-ignored), samples/sample_batch.csv
├── notebooks/                  01_eda_and_training.ipynb
├── reports/                    metrics.json, CSVs, figures/, Task9_10_Report.pdf
├── docs/                       SRS, SDD, USER_MANUAL, RESPONSIBLE_AI_REPORT, DEPLOYMENT_GUIDE, TESTING_REPORT, FINAL_REPORT, screenshots/
└── tests/
```

## Limitations
- Only {n_fraud_test} frauds in the test set; metrics carry wide uncertainty.
- V1-V28 are anonymised PCA components, so SHAP factors are not business-readable and protected-attribute proxies cannot be ruled out.
- Two days of 2013 data with a random split; re-validate on recent traffic before real use.
- No customer attributes, so fairness is approximated with amount and time-of-day segments.
- Docker image and Hugging Face deployment were documented but not executed in the build environment.
"""
(ROOT / "README.md").write_text(w_readme.strip() + "\n"); print("wrote README.md")
