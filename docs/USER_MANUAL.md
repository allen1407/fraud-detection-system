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
