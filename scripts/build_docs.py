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
fg, fa = pd.read_csv(R / "fairness_gender.csv"), pd.read_csv(R / "fairness_age.csv")
drift, ready, tests = pd.read_csv(R / "drift_baseline.csv"), pd.read_csv(R / "deployment_readiness.csv"), pd.read_csv(R / "testing_report.csv")
rb, shg = pd.read_csv(R / "robustness.csv"), pd.read_csv(R / "shap_global.csv")
d, l = m["dnn"], m["lr"]
cmd, cml = d["confusion_matrix"], l["confusion_matrix"]
n_fail = int((ready.status == "FAIL").sum())
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
| Flag rate | {d['flag_rate']:.1%} | {l['flag_rate']:.1%} |
| Confusion matrix (TN/FP/FN/TP) | {cmd['tn']}/{cmd['fp']}/{cmd['fn']}/{cmd['tp']} | {cml['tn']}/{cml['fp']}/{cml['fn']}/{cml['tp']} |

Test set: {m['n_test']} transactions ({m['test_fraud_rate']:.1%} fraud). Source: `reports/metrics.json`."""

HONESTY = f"""> **Key finding.** Both models score close to chance (test ROC-AUC {d['roc_auc']:.3f} DNN, {l['roc_auc']:.3f} LR). A 5-fold gradient-boosting check on the raw data gives ≈0.51 as well. The dataset shows no learnable fraud signal (fraud rate is ≈49% in every category, and the account numbers and descriptions are unique per row, which suggests simulated data). The system demonstrates integration, explainability, testing and governance; it does **not** demonstrate real fraud-detection skill."""

THRESH = f"""The operating DNN threshold is **{meta['threshold']:.2f}**: the F1-maximising threshold on the validation split *subject to a validation false-positive-rate guardrail of {meta['max_val_fpr_guardrail']:.0%}*. Without the guardrail, F1 is maximised at threshold {meta['threshold_f1_unconstrained']:.2f}, which flags 100% of transactions (F1 = 0.66 simply because the base rate is ≈49%), which is the "flag everything" failure seen in an earlier version of this project. The cost-optimal validation threshold is also {meta['threshold_cost_optimal']:.2f} (flag everything costs ${cost['flag_all_cost']:,.0f} vs ${cost['approve_all_cost']:,.0f} for approving everything, because a model with no signal cannot beat flagging everything when a missed fraud costs $500 and a false alarm $15). On the test set the cost optimum is {cost['optimum_threshold']:.2f} (${cost['optimum_cost']:,.0f}). All three thresholds are recorded in `models/metadata.json`."""


def w(name, text):
    (D / name).write_text(text.strip() + "\n"); print("wrote", name)


w("SRS.md", f"""
# Software Requirements Specification — Intelligent Fraud Detection System

## 1. Purpose and scope
A decision-support system that validates a banking transaction, scores it with two models, explains the result and recommends AUTO-BLOCK, AUTO-APPROVE or MANUAL REVIEW. Scope: Task 9 (integration & deployment) and Task 10 (evaluation, responsible AI, documentation). Training data: {data['rows']:,} transactions ({data['fraud_rate']:.1%} fraud), {data['date_min'][:10]} to {data['date_max'][:10]}.

## 2. Functional requirements
| ID | Requirement | Implemented in | Verified by |
|---|---|---|---|
| FR1 | Validate input: required fields, no non-positive amount, valid age/income/balance ranges, known categories; return a list of errors | `FraudPipeline.validate` | `test_negative_amount_rejected`, `test_missing_field_rejected`, `test_out_of_range_and_unknown_category_rejected` |
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
| NFR7 | Privacy | no raw customer data persisted by the app; reports are generated on demand | yes |
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
| `src/config.py` | paths, seed 42, costs, schema, ranges, verdict strings |
| `src/data.py` | load/clean, stratified 70/15/15 split ({meta['split_sizes']['train']}/{meta['split_sizes']['val']}/{meta['split_sizes']['test']}) |
| `src/features.py` | `FeatureBuilder` (hour, day of week, month, weekend, Amount_to_Income, Amount_to_Balance, occupation and merchant frequency) + `ColumnTransformer` (StandardScaler + OneHotEncoder), fit on train only |
| `src/train.py` | LR + DNN (64-32-16, dropout 0.3, L2, early stopping on validation loss), threshold selection, sanity gate |
| `src/pipeline.py` | `FraudPipeline`: validate, score, explain, trust_score, route, generate_report, score_batch |
| `src/evaluate.py` | all metrics, figures, fairness, robustness, drift, latency, SHAP stability, readiness |
| `src/ui.py`, `app.py`, `pages/` | Streamlit UI (cached resources, presets, charts); calls only `src/` |
| `scripts/` | `prepare_data.py`, `run_all.sh`, `build_docs.py`, `build_report_pdf.py`, `screenshots.py` |

## 3. Inference data flow
1. `validate(record)` returns a list of errors (empty when valid).
2. `FeatureBuilder` derives the {len(meta['feature_list'])}-column model input (after one-hot encoding) with the saved preprocessor.
3. DNN and LR produce probabilities; the DNN is compared with threshold {meta['threshold']:.2f}, LR with 0.50.
4. Trust = (95th percentile of leave-one-out mean 5-NN distances in the training set) ÷ (mean 5-NN distance of the query), capped at 10. Values ≥ 1.0 mean the transaction is as familiar as 95% of training rows.
5. `route` combines agreement and trust. Explanation uses `shap.KernelExplainer` on the 16 engineered features (12 numeric, 4 categorical; categories decoded before the preprocessor) with a 100-row background sample built once.

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
2. Fill in the fields (date, time, amount, age, income, balance, type, merchant category, payment method, gender, occupation, merchant name).
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

## 1. Fairness
Computed for the DNN at threshold {d['threshold']:.2f} on the test set (`reports/fairness_gender.csv`, `reports/fairness_age.csv`).

**By gender**

{md(fg.round(3))}

**By age band**

{md(fa.round(3))}

Largest flag-rate gaps: gender {fg.iloc[-1].flag_rate:.3f}, age {fa.iloc[-1].flag_rate:.3f}; both exceed the 0.10 audit criterion. With roughly 110-300 rows per group and a model near chance, these differences are largely sampling noise, but they cannot be dismissed. On real data they would require investigation (and, if confirmed, mitigation) before deployment. Gender is a model input here; a production system should reconsider using protected attributes.

## 2. Robustness
Counterfactual search over the four numeric inputs (15 test cases, 400 random directions per magnitude, magnitudes up to 3 training standard deviations; `reports/robustness.csv`). The DNN class flipped in {rob['dnn_flip_rate']:.0%} of cases, the routed verdict in {rob['verdict_flip_rate']:.0%}; the median distance to a flip is {rob['median_distance']:.2f} standard deviations. Because the DNN outputs sit close to the 0.5 threshold (they carry little information), small perturbations are enough to flip outcomes; this is a symptom of the weak signal rather than evidence of a sharp decision boundary.

## 3. Explainability
SHAP (KernelExplainer, 100-row background) gives the top-5 factors per decision. Stability: mean Spearman rank correlation {st['mean_spearman']:.3f} ± {st['std_spearman']:.3f} (min {st['min_spearman']:.3f}) under 1% numeric noise over {st['n_samples']} samples (`reports/shap_stability.json`). Global importance (mean |SHAP|, top 5): {', '.join(f"{r.feature} ({r.mean_abs_shap:.3f})" for r in shg.head(5).itertuples())}. Stable explanations are not the same as meaningful ones: with no real signal, the attributed factors reflect noise the network fit.

## 4. Privacy
The app stores no transactions: inputs live only in the session, reports are generated on demand, and nothing is sent to external services. The data contains account numbers and customer attributes; they are excluded from model features (IDs, descriptions) but remain in `data/`. Deploy with access control and do not commit real customer data.

## 5. Transparency
Every decision comes with both probabilities, model agreement, trust score, SHAP factors and a plain-language recommendation. `models/metadata.json` records features, thresholds, library versions, training date and dataset size. Known weaknesses are shown in the app (Monitoring page and Documentation).

## 6. Human oversight
The system never acts alone on uncertain cases. Automatic action requires both models to agree **and** trust ≥ 1.0. On the test set, {m['routing'].get('MANUAL REVIEW', 0)} of {m['n_test']} transactions ({m['routing'].get('MANUAL REVIEW', 0) / m['n_test']:.0%}) were routed to MANUAL REVIEW, {m['routing'].get('AUTO-APPROVE', 0)} auto-approved and {m['routing'].get('AUTO-BLOCK', 0)} auto-blocked. Because the models are near chance, **auto-actions should not be enabled in practice**; every decision should be human-reviewed.

## 7. Threshold choice
{THRESH}

## 8. Limitations
- The dataset has no learnable signal, so metrics near 0.5 AUC are expected and no real fraud-detection capability is demonstrated.
- A ≈49% fraud rate is unrealistic for live banking traffic (typically well under 1%). Precision, flag rate and the cost trade-off would be very different; thresholds must be recalibrated on realistic prevalence before any real use.
- The data appears synthetic, so findings may not transfer to real transactions.
- Fairness estimates are noisy (small groups) and only cover gender and age band.
- Drift baseline compares train with a random test split from the same period; it says nothing about future drift.
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
The image contains the trained models and reports committed to the repo. Rebuild after retraining.

## Hugging Face Spaces
1. Create a Space with SDK **Streamlit** (or **Docker** and reuse the Dockerfile with `app_port: 8501`).
2. Push the repo (including `models/`, `reports/`, `data/processed/train.csv`, `data/processed/test.csv`; the app needs them at runtime).
3. Set the Python version to 3.11 and keep `requirements.txt` pinned.
Do not host real customer data on a public Space.

## Monitoring
- **Monthly:** compute PSI of each numeric feature against the training distribution (`src/evaluate.py` shows the method; baseline in `reports/drift_baseline.csv`, max PSI {drift.psi.max():.3f} = {drift.sort_values('psi').iloc[-1].feature}).
- **Retrain trigger:** any PSI > 0.25.
- **Schedule:** retrain quarterly regardless.
- Track the flag rate, the MANUAL REVIEW share and analyst-confirmed fraud rate.

## Rollback
Keep the previous `fraud_lr_model.pkl`, `fraud_dnn_model.keras`, `preprocessor.pkl` and `metadata.json` together in a dated folder (e.g. `models/archive/YYYY-MM-DD/`). To roll back, restore the whole set at once; never mix a model with a different preprocessor.

## Readiness status
{md(dep)}

**{n_fail} of {len(ready)} checks fail.** Predictive power and fairness criteria are not met, so this model must not be used to take automatic decisions on real transactions.
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
`FraudPipeline` (`src/pipeline.py`) integrates validation, feature engineering, two models (Keras DNN, logistic regression), SHAP explanations, a nearest-neighbour trust score, human-in-the-loop routing and report generation. A Streamlit app (`app.py`, `pages/`) exposes it through six pages and only calls `src/`. Dataset: {data['rows']:,} transactions, {data['fraud_rate']:.1%} fraud; split 70/15/15 stratified ({meta['split_sizes']['train']}/{meta['split_sizes']['val']}/{meta['split_sizes']['test']}); preprocessor fit on train only.

{HONESTY}

## 2. Deliverables inventory
| Deliverable | Path |
|---|---|
| Pipeline | `src/pipeline.py` |
| Features / data / training / evaluation | `src/features.py`, `src/data.py`, `src/train.py`, `src/evaluate.py` |
| App | `app.py`, `pages/`, `src/ui.py`, `.streamlit/config.toml` |
| Models | `models/fraud_lr_model.pkl`, `models/fraud_dnn_model.keras`, `models/preprocessor.pkl`, `models/metadata.json` |
| Metrics and CSVs | `reports/metrics.json`, `cost_threshold.csv`, `latency.json`, `shap_stability.json`, `fairness_gender.csv`, `fairness_age.csv`, `robustness.csv`, `drift_baseline.csv`, `deployment_readiness.csv`, `testing_report.csv` |
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
![Confusion matrices](../reports/figures/confusion_matrix.png)

{THRESH}

![Cost vs threshold](../reports/figures/cost_threshold_analysis.png)

Latency: single row {lat['single_no_explanation_ms']['mean']:.1f} ms mean / {lat['single_no_explanation_ms']['p95']:.1f} ms p95 without SHAP; {lat['single_with_shap_ms']['mean']:.0f} ms mean / {lat['single_with_shap_ms']['p95']:.0f} ms p95 with SHAP; batch {lat['batch']['rows_per_second']:,.0f} rows/s.

## 5. Explainability
Top-5 SHAP factors per decision; stability {st['mean_spearman']:.3f} ± {st['std_spearman']:.3f} Spearman under 1% noise. Most important features: {', '.join(shg.head(5).feature)}.

![SHAP](../reports/figures/shap_summary.png)

## 6. Responsible AI
Fairness gaps (flag rate): gender {fg.iloc[-1].flag_rate:.3f}, age {fa.iloc[-1].flag_rate:.3f} (criterion 0.10; not met, noisy). Robustness: DNN flips in {rob['dnn_flip_rate']:.0%} of 15 cases, median distance {rob['median_distance']:.2f} SD. Human review: {m['routing'].get('MANUAL REVIEW', 0)} of {m['n_test']} test transactions go to MANUAL REVIEW. Full discussion in `docs/RESPONSIBLE_AI_REPORT.md`.

## 7. Deployment readiness
{md(ready)}

**{n_fail} of {len(ready)} checks fail** (predictive power, cost vs flag-all, fairness). The software checks (latency, throughput, stability, drift, tests, artifacts) pass.

## 8. Limitations
No learnable signal in the data; unrealistic ≈49% fraud prevalence (thresholds need recalibration on realistic traffic); likely synthetic data; small fairness groups; train-vs-test drift only; SHAP explains the DNN only.

## 9. Future enhancements
Use real labelled data with realistic prevalence; recalibrate thresholds and probabilities; add temporal validation; richer behavioural features (velocity, device, geography); fairness mitigation; scheduled PSI monitoring job and an analyst feedback loop; authentication and audit logging.
""")

# README results block

# ---------------------------------------------------------------- README
w_readme = f"""
# Intelligent Fraud Detection System

A Streamlit decision-support system that scores banking transactions with a Keras DNN and logistic regression, explains them with SHAP, and routes uncertain cases to human review (Computational Intelligence Lab, Tasks 9 and 10).

{HONESTY}

## Dataset
Kaggle banking-transactions fraud dataset (USA, 2023-2024). **{data['rows']:,} rows**, **{data['fraud_rate']:.1%} fraud** ({data['fraud_rows']:,} fraud rows), {data['date_min'][:10]} to {data['date_max'][:10]}, no missing values. Columns used: transaction date/time, amount, merchant, merchant category, transaction type, payment method, customer age, gender, occupation, income, account balance; label `Fraud_Flag`.

Source: [Kaggle link]

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
| Figures (ROC, PR, confusion, cost, SHAP, fairness, drift) | `reports/figures/` | Done |
| Cost sweep | `reports/cost_threshold.csv` | Done |
| Latency | `reports/latency.json` | Done |
| SHAP stability | `reports/shap_stability.json` | Done |
| Fairness | `reports/fairness_gender.csv`, `reports/fairness_age.csv` | Done |
| Robustness | `reports/robustness.csv` | Done |
| Drift baseline | `reports/drift_baseline.csv` | Done |
| Readiness checklist | `reports/deployment_readiness.csv` | Done ({n_fail} checks FAIL) |
| Tests + report | `tests/`, `reports/testing_report.csv` | Done ({int((tests.result == 'PASSED').sum())}/{len(tests)} pass) |
| SRS, SDD, user manual, RAI report, testing report, final report | `docs/` | Done |
| Final report PDF | `reports/Task9_10_Report.pdf` | Done |
| Notebook (executed) | `notebooks/01_eda_and_training.ipynb` | Done |

## Headline results (test set)
{HEADLINE}

Operating threshold: {meta['threshold']:.2f}, see the threshold note in `docs/SDD.md`. Readiness: {n_fail} of {len(ready)} checks FAIL (predictive power, cost vs flag-all, fairness gaps); do not use for real automatic decisions.

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
# 1. place the dataset zip in data/raw/
python scripts/prepare_data.py data/raw/archive.zip   # 2. prepare data
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
├── data/                       raw/ (zip + csv), processed/, samples/sample_batch.csv
├── notebooks/                  01_eda_and_training.ipynb
├── reports/                    metrics.json, CSVs, figures/, Task9_10_Report.pdf
├── docs/                       SRS, SDD, USER_MANUAL, RESPONSIBLE_AI_REPORT, DEPLOYMENT_GUIDE, TESTING_REPORT, FINAL_REPORT, screenshots/
└── tests/
```

## Limitations
- No learnable fraud signal in this dataset (ROC-AUC ≈ 0.5); the system shows integration, not detection skill.
- ≈49% fraud rate is unrealistic for live banking traffic; thresholds must be recalibrated before real use.
- The data looks synthetic; fairness groups are small; SHAP explains the DNN only.
- Docker image and Hugging Face deployment were documented but not executed in the build environment.
- Raw data licence not verified: check the Kaggle licence before redistributing `data/raw/`.
"""
(ROOT / "README.md").write_text(w_readme.strip() + "\n"); print("wrote README.md")
