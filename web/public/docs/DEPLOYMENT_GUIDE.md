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
- **Monthly:** compute PSI of each model feature against the training distribution (`src/evaluate.py` shows the method; baseline in `reports/drift_baseline.csv`, max PSI 0.001 = V8).
- **Retrain trigger:** any PSI > 0.25.
- **Schedule:** retrain quarterly regardless.
- Track the flag rate, the MANUAL REVIEW share and analyst-confirmed fraud rate.

## Rollback
Keep the previous `fraud_lr_model.pkl`, `fraud_dnn_model.keras`, `preprocessor.pkl` and `metadata.json` together in a dated folder (e.g. `models/archive/YYYY-MM-DD/`). To roll back, restore the whole set at once; never mix a model with a different preprocessor.

## Readiness status
| check                              | criterion                            | measured                           | status   |
|:-----------------------------------|:-------------------------------------|:-----------------------------------|:---------|
| DNN predictive power               | test ROC-AUC >= 0.70                 | 0.951                              | PASS     |
| LR predictive power                | test ROC-AUC >= 0.70                 | 0.969                              | PASS     |
| DNN precision-recall (rare class)  | test PR-AUC >= 0.50                  | 0.711                              | PASS     |
| Flag rate sanity                   | DNN flag rate <= 50%                 | 0.2%                               | PASS     |
| False-positive rate                | DNN FPR <= 30%                       | 0.0%                               | PASS     |
| Expected cost vs flag-all baseline | DNN cost < flag-everything cost      | $7,695 vs $637,320                 | PASS     |
| Single-row latency (no SHAP)       | p95 <= 100 ms                        | 10.0 ms                            | PASS     |
| Single-row latency (with SHAP)     | p95 <= 2000 ms                       | 545 ms                             | PASS     |
| Batch throughput                   | >= 100 rows/s                        | 4197 rows/s                        | PASS     |
| Explanation stability              | mean Spearman >= 0.80                | 0.995 +/- 0.019                    | PASS     |
| Subgroup recall (amount band)      | recall gap <= 0.20                   | 0.311                              | FAIL     |
| Subgroup recall (time of day)      | recall gap <= 0.20                   | 0.265                              | FAIL     |
| Robustness                         | median flip distance >= 0.5 std      | >= 3.000 std (15/15 never flipped) | PASS     |
| Data drift (train vs test)         | max PSI <= 0.25                      | 0.001                              | PASS     |
| Automated tests                    | all pass                             | 13/13 passed                       | PASS     |
| Model artifacts & metadata         | all files present, metadata complete | complete                           | PASS     |

**2 of 16 checks fail** (Subgroup recall (amount band), Subgroup recall (time of day)). The predictive, cost, latency and engineering checks pass; the subgroup recall gaps rest on a handful of frauds per segment and must be re-checked (or formally accepted) before production. Independently of the checklist, the model must be re-validated on recent traffic before any real use.
