# Responsible AI Report

> **Key finding.** Both models detect fraud well in ranking terms: test ROC-AUC 0.951 (DNN) and 0.969 (LR), PR-AUC 0.711 and 0.725 against a fraud base rate of only 0.17%. At its operating threshold the DNN catches 56 of 71 test frauds with 13 false alarms among 42,488 legitimate transactions. The simple logistic regression ranks slightly better than the DNN; the DNN's advantage is higher recall at the chosen threshold. Caveats: only 71 frauds in the test set (wide uncertainty), anonymised PCA features (explanations are not business-readable), two days of 2013 data, a random rather than temporal split, and no customer attributes for a fairness audit.

## 1. Fairness and subgroup performance
The dataset contains **no customer attributes** (no gender, age, nationality or location), so fairness across protected groups cannot be measured, and the model cannot use protected attributes directly. Because V1-V28 are PCA mixtures of undisclosed original features, proxies for protected attributes may still be embedded in them; this cannot be checked without the original data. As a substitute, performance is compared across transaction segments for the DNN at threshold 0.40 on the test set (`reports/fairness_amount.csv`, `reports/fairness_time.csv`).

**By amount band**

| group    |     n |   n_fraud |   flag_rate |    tpr |    fpr |   precision |
|:---------|------:|----------:|------------:|-------:|-------:|------------:|
| <=$10    | 14893 |        34 |      0.0021 | 0.7647 | 0.0003 |      0.8387 |
| $10-100  | 19156 |        17 |      0.001  | 0.8824 | 0.0003 |      0.75   |
| $100-500 |  7129 |        13 |      0.0018 | 0.8462 | 0.0003 |      0.8462 |
| >$500    |  1381 |         7 |      0.0036 | 0.5714 | 0.0007 |      0.8    |
| MAX GAP  | 42559 |        71 |      0.0026 | 0.3109 | 0.0005 |      0.0962 |

**By time of day**

| group         |     n |   n_fraud |   flag_rate |    tpr |    fpr |   precision |
|:--------------|------:|----------:|------------:|-------:|-------:|------------:|
| Night 00-06   |  3600 |        20 |      0.0061 | 0.85   | 0.0014 |      0.7727 |
| Day 06-18     | 25121 |        34 |      0.0014 | 0.8529 | 0.0003 |      0.8056 |
| Evening 18-24 | 13838 |        17 |      0.0008 | 0.5882 | 0.0001 |      0.9091 |
| MAX GAP       | 42559 |        71 |      0.0053 | 0.2647 | 0.0013 |      0.1364 |

Largest recall (TPR) gaps: amount band 0.311, time of day 0.265 (criterion 0.20). Each segment holds only 7-34 frauds, so one or two transactions move recall by 0.1 or more; the gaps are not statistically conclusive but are flagged for re-checking on more data. False-positive rates are below 0.2% in every segment.

## 2. Robustness
Counterfactual search over V1-V28 and Amount for 15 test cases (5 fraud, 10 legitimate), 400 random directions per magnitude, magnitudes up to 3 training standard deviations (`reports/robustness.csv`). The DNN class flipped in 0% of cases and the routed verdict in 0%; 15 of 15 cases never flipped inside the search radius (median distance reported as ≥ 3.0 SD). This shows the decisions are not sensitive to random noise. It does **not** show adversarial robustness: random directions in 29 dimensions rarely point at the decision boundary, and a gradient-based attacker would need far less. Note also that 4 of the 5 sampled frauds were already missed by the model (probability below the threshold); random perturbation leaves them missed.

## 3. Explainability
SHAP (KernelExplainer, 100-row background) gives the top-5 factors per decision. Stability: mean Spearman rank correlation 0.995 ± 0.019 (min 0.892) under 1% noise on V1-V28 and Amount over 30 samples (`reports/shap_stability.json`). Global importance (mean |SHAP|, top 5): V14 (1.99e-04), V4 (1.73e-04), V12 (1.32e-04), V10 (6.64e-05), V16 (5.45e-05). Mean |SHAP| values are tiny because almost every transaction scores near 0; for flagged transactions the attributions are orders of magnitude larger. The top SHAP components (V14, V12, V10) are also among the strongest class separators in the EDA notebook, so the explanations are consistent with the data. They are, however, **not business-interpretable**: "V14 is low" cannot be explained to a customer or mapped to a rule without the confidential PCA loadings.

## 4. Privacy
The app stores no transactions: inputs live only in the session, reports are generated on demand, and nothing is sent to external services. The dataset was anonymised by the provider (PCA transformation, no identifiers, no customer attributes), so it carries little re-identification risk. A production system fed with raw card data would need access control, encryption and PCI-DSS compliance. The raw and processed CSVs are excluded from git (`.gitignore`) because of their size.

## 5. Transparency
Every decision comes with both probabilities, model agreement, trust score, SHAP factors and a plain-language recommendation. `models/metadata.json` records features, thresholds, library versions, training date and dataset size. Known weaknesses are shown in the app (Monitoring page and Documentation).

## 6. Human oversight
The system never acts alone on uncertain cases. Automatic action requires both models to agree **and** trust ≥ 1.0. On the test set, 1,895 of 42,559 transactions (4.5%) were routed to MANUAL REVIEW, 40,649 auto-approved and 15 auto-blocked. Low trust drives the queue: 1,890 review cases have trust < 1.0 (unusual transactions), while the models disagree on only 17 test transactions. Of the 71 test frauds, 14 were auto-blocked, 45 reached an analyst and 12 were auto-approved (missed). Of the frauds sent to review, 33 were flagged by both models but held back because they look unlike the reference data (frauds are unusual by nature), 9 were flagged by one model only, and 3 were flagged by neither and reached an analyst only because of low trust. The price is an analyst queue of about 4.5% of traffic, almost all legitimate; `TRUST_MIN` in `src/config.py` controls this trade-off.

## 7. Threshold choice
The operating DNN threshold is **0.40**: the F1-maximising threshold on the validation split, subject to a validation false-positive-rate guardrail of 30% (the guardrail is not binding here; the unconstrained F1 optimum is also 0.40). With a missed fraud costing $500 and a false alarm $15, the cost-optimal validation threshold is lower, **0.04**. On the test set it catches 58 frauds with 34 false alarms (cost $7,010), against 56 frauds and 13 false alarms (cost $7,695) at 0.40. Baselines: approving everything costs $35,500; flagging everything $637,320. The F1 threshold is kept as the operating point because it keeps the alert volume lowest; a bank that prices misses at $500 should consider switching to 0.04. All thresholds are recorded in `models/metadata.json`.

## 8. Limitations
- Only 473 frauds in total (71 in the test set): recall and precision carry wide confidence intervals (one fraud moves test recall by 1.4%).
- Features are anonymised PCA components: explanations cannot be given to customers in plain language, and proxies for protected attributes cannot be ruled out.
- Two days of 2013 European card traffic; fraud patterns change quickly, so performance on current traffic is unknown.
- Random (not temporal) split: train and test overlap in time, so metrics and the drift baseline are optimistic.
- No customer attributes: fairness is approximated with transaction segments only.
- SHAP explains the DNN only and is an approximation.
