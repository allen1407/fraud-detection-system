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
- **Monthly:** compute PSI of each numeric feature against the training distribution (`src/evaluate.py` shows the method; baseline in `reports/drift_baseline.csv`, max PSI 0.161 = occupation_freq).
- **Retrain trigger:** any PSI > 0.25.
- **Schedule:** retrain quarterly regardless.
- Track the flag rate, the MANUAL REVIEW share and analyst-confirmed fraud rate.

## Rollback
Keep the previous `fraud_lr_model.pkl`, `fraud_dnn_model.keras`, `preprocessor.pkl` and `metadata.json` together in a dated folder (e.g. `models/archive/YYYY-MM-DD/`). To roll back, restore the whole set at once; never mix a model with a different preprocessor.

## Readiness status
| check                              | criterion                            | measured           | status   |
|:-----------------------------------|:-------------------------------------|:-------------------|:---------|
| DNN predictive power               | test ROC-AUC >= 0.70                 | 0.524              | FAIL     |
| LR predictive power                | test ROC-AUC >= 0.70                 | 0.510              | FAIL     |
| Flag rate sanity                   | DNN flag rate <= 50%                 | 28.2%              | PASS     |
| False-positive rate                | DNN FPR <= 30%                       | 26.8%              | PASS     |
| Expected cost vs flag-all baseline | DNN cost < flag-everything cost      | $142,150 vs $6,150 | FAIL     |
| Single-row latency (no SHAP)       | p95 <= 100 ms                        | 5.8 ms             | PASS     |
| Single-row latency (with SHAP)     | p95 <= 2000 ms                       | 123 ms             | PASS     |
| Batch throughput                   | >= 100 rows/s                        | 14159 rows/s       | PASS     |
| Explanation stability              | mean Spearman >= 0.80                | 0.997 +/- 0.005    | PASS     |
| Fairness (gender)                  | flag-rate gap <= 0.10                | 0.123              | FAIL     |
| Fairness (age band)                | flag-rate gap <= 0.10                | 0.187              | FAIL     |
| Robustness                         | median flip distance >= 0.5 std      | 1.124 std          | PASS     |
| Data drift (train vs test)         | max PSI <= 0.25                      | 0.161              | PASS     |
| Automated tests                    | all pass                             | 13/13 passed       | PASS     |
| Model artifacts & metadata         | all files present, metadata complete | complete           | PASS     |

**5 of 15 checks fail.** Predictive power and fairness criteria are not met, so this model must not be used to take automatic decisions on real transactions.
