# Software Requirements Specification — Intelligent Fraud Detection System

## 1. Purpose and scope
A decision-support system that validates a banking transaction, scores it with two models, explains the result and recommends AUTO-BLOCK, AUTO-APPROVE or MANUAL REVIEW. Scope: Task 9 (integration & deployment) and Task 10 (evaluation, responsible AI, documentation). Training data: 5,389 transactions (49.3% fraud), 2023-01-21 to 2025-01-20.

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
| NFR1 | Single-row latency (no SHAP) | p95 ≤ 100 ms | 5.8 ms |
| NFR2 | Single-row latency (with SHAP) | p95 ≤ 2000 ms | 123 ms |
| NFR3 | Batch throughput | ≥ 100 rows/s | 14,159 rows/s |
| NFR4 | Explanation stability under 1% noise | mean Spearman ≥ 0.80 | 0.997 |
| NFR5 | Reproducibility | fixed seed 42, pinned `requirements.txt`, `scripts/run_all.sh` | yes |
| NFR6 | Usability | friendly errors, no stack traces in the UI | yes |
| NFR7 | Privacy | no raw customer data persisted by the app; reports are generated on demand | yes |
| NFR8 | Portability | local, Docker, Hugging Face Spaces | see DEPLOYMENT_GUIDE |

## 4. Constraints and assumptions
Cost assumptions: false negative $500, false positive $15. Python 3.11, TensorFlow 2.16, scikit-learn 1.5.

> **Key finding.** Both models score close to chance (test ROC-AUC 0.524 DNN, 0.510 LR). A 5-fold gradient-boosting check on the raw data gives ≈0.51 as well. The dataset shows no learnable fraud signal (fraud rate is ≈49% in every category, and the account numbers and descriptions are unique per row, which suggests simulated data). The system demonstrates integration, explainability, testing and governance; it does **not** demonstrate real fraud-detection skill.
