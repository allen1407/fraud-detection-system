# Testing Report

Run: `python -m pytest -q`; results exported to `reports/testing_report.csv`. 13/13 tests passed.

| test                                            | file                   | result   |   duration_s |
|:------------------------------------------------|:-----------------------|:---------|-------------:|
| test_valid_transaction_scores                   | tests/test_pipeline.py | PASSED   |        0.084 |
| test_negative_amount_rejected                   | tests/test_pipeline.py | PASSED   |        0     |
| test_missing_field_rejected                     | tests/test_pipeline.py | PASSED   |        0     |
| test_out_of_range_and_unknown_category_rejected | tests/test_pipeline.py | PASSED   |        0     |
| test_probabilities_in_unit_interval             | tests/test_pipeline.py | PASSED   |        0.084 |
| test_recommendation_in_allowed_set              | tests/test_pipeline.py | PASSED   |        0.183 |
| test_report_nonempty_for_valid_and_invalid      | tests/test_pipeline.py | PASSED   |        0.069 |
| test_trust_score_positive                       | tests/test_pipeline.py | PASSED   |        0.01  |
| test_routing_rules                              | tests/test_pipeline.py | PASSED   |        0     |
| test_batch_of_20_runs_with_invalid_rows         | tests/test_pipeline.py | PASSED   |        0.027 |
| test_batch_vs_single_parity                     | tests/test_pipeline.py | PASSED   |        0.292 |
| test_latency_under_criterion                    | tests/test_pipeline.py | PASSED   |        0.099 |
| test_shap_top5_has_direction_and_sign           | tests/test_pipeline.py | PASSED   |        0.057 |

Latency criterion: p95 single-row latency (no SHAP) < 100 ms; measured by `src/evaluate.py` at 5.8 ms (`reports/latency.json`).
