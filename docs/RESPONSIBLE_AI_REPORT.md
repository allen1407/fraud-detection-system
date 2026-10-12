# Responsible AI Report

> **Key finding.** Both models score close to chance (test ROC-AUC 0.524 DNN, 0.510 LR). A 5-fold gradient-boosting check on the raw data gives ≈0.51 as well. The dataset shows no learnable fraud signal (fraud rate is ≈49% in every category, and the account numbers and descriptions are unique per row, which suggests simulated data). The system demonstrates integration, explainability, testing and governance; it does **not** demonstrate real fraud-detection skill.

## 1. Fairness
Computed for the DNN at threshold 0.50 on the test set (`reports/fairness_gender.csv`, `reports/fairness_age.csv`).

**By gender**

| group   |   n |   flag_rate |   tpr |   fpr |   precision |
|:--------|----:|------------:|------:|------:|------------:|
| Female  | 269 |       0.283 | 0.307 | 0.261 |       0.513 |
| Male    | 268 |       0.343 | 0.41  | 0.276 |       0.598 |
| Others  | 272 |       0.221 | 0.174 | 0.269 |       0.4   |
| MAX GAP | 809 |       0.123 | 0.237 | 0.016 |       0.198 |

**By age band**

| group   |   n |   flag_rate |   tpr |   fpr |   precision |
|:--------|----:|------------:|------:|------:|------------:|
| <=25    | 111 |       0.36  | 0.391 | 0.319 |       0.625 |
| 26-40   | 234 |       0.363 | 0.368 | 0.358 |       0.494 |
| 41-60   | 305 |       0.246 | 0.241 | 0.25  |       0.467 |
| 60+     | 159 |       0.176 | 0.211 | 0.145 |       0.571 |
| MAX GAP | 809 |       0.187 | 0.18  | 0.214 |       0.158 |

Largest flag-rate gaps: gender 0.123, age 0.187; both exceed the 0.10 audit criterion. With roughly 110-300 rows per group and a model near chance, these differences are largely sampling noise, but they cannot be dismissed. On real data they would require investigation (and, if confirmed, mitigation) before deployment. Gender is a model input here; a production system should reconsider using protected attributes.

## 2. Robustness
Counterfactual search over the four numeric inputs (15 test cases, 400 random directions per magnitude, magnitudes up to 3 training standard deviations; `reports/robustness.csv`). The DNN class flipped in 100% of cases, the routed verdict in 67%; the median distance to a flip is 1.12 standard deviations. Because the DNN outputs sit close to the 0.5 threshold (they carry little information), small perturbations are enough to flip outcomes; this is a symptom of the weak signal rather than evidence of a sharp decision boundary.

## 3. Explainability
SHAP (KernelExplainer, 100-row background) gives the top-5 factors per decision. Stability: mean Spearman rank correlation 0.997 ± 0.005 (min 0.979) under 1% numeric noise over 30 samples (`reports/shap_stability.json`). Global importance (mean |SHAP|, top 5): Account_Balance (0.026), hour (0.018), Customer_Age (0.014), Category (0.013), month (0.010). Stable explanations are not the same as meaningful ones: with no real signal, the attributed factors reflect noise the network fit.

## 4. Privacy
The app stores no transactions: inputs live only in the session, reports are generated on demand, and nothing is sent to external services. The data contains account numbers and customer attributes; they are excluded from model features (IDs, descriptions) but remain in `data/`. Deploy with access control and do not commit real customer data.

## 5. Transparency
Every decision comes with both probabilities, model agreement, trust score, SHAP factors and a plain-language recommendation. `models/metadata.json` records features, thresholds, library versions, training date and dataset size. Known weaknesses are shown in the app (Monitoring page and Documentation).

## 6. Human oversight
The system never acts alone on uncertain cases. Automatic action requires both models to agree **and** trust ≥ 1.0. On the test set, 267 of 809 transactions (33%) were routed to MANUAL REVIEW, 400 auto-approved and 142 auto-blocked. Because the models are near chance, **auto-actions should not be enabled in practice**; every decision should be human-reviewed.

## 7. Threshold choice
The operating DNN threshold is **0.50**: the F1-maximising threshold on the validation split *subject to a validation false-positive-rate guardrail of 30%*. Without the guardrail, F1 is maximised at threshold 0.01, which flags 100% of transactions (F1 = 0.66 simply because the base rate is ≈49%), which is the "flag everything" failure seen in an earlier version of this project. The cost-optimal validation threshold is also 0.01 (flag everything costs $6,150 vs $199,500 for approving everything, because a model with no signal cannot beat flagging everything when a missed fraud costs $500 and a false alarm $15). On the test set the cost optimum is 0.25 ($6,120). All three thresholds are recorded in `models/metadata.json`.

## 8. Limitations
- The dataset has no learnable signal, so metrics near 0.5 AUC are expected and no real fraud-detection capability is demonstrated.
- A ≈49% fraud rate is unrealistic for live banking traffic (typically well under 1%). Precision, flag rate and the cost trade-off would be very different; thresholds must be recalibrated on realistic prevalence before any real use.
- The data appears synthetic, so findings may not transfer to real transactions.
- Fairness estimates are noisy (small groups) and only cover gender and age band.
- Drift baseline compares train with a random test split from the same period; it says nothing about future drift.
- SHAP explains the DNN only and is an approximation.
