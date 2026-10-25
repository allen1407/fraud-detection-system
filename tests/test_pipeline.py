import time

import numpy as np
import pandas as pd

from src import config as C


def test_valid_transaction_scores(pipe, valid_record):
    r = pipe.assess(valid_record)
    assert r["valid"] and r["verdict"] in ("AUTO-BLOCK", "AUTO-APPROVE", "MANUAL REVIEW")
    assert len(r["factors"]) == 5


def test_negative_amount_rejected(pipe, valid_record):
    errs = pipe.validate(dict(valid_record, Amount=-10))
    assert any("Amount" in e for e in errs)
    assert pipe.assess(dict(valid_record, Amount=-10))["verdict"] == "TRANSACTION REJECTED"


def test_missing_field_rejected(pipe, valid_record):
    rec = dict(valid_record); rec.pop("V14")
    assert any("V14" in e for e in pipe.validate(rec))


def test_out_of_range_and_non_numeric_rejected(pipe, valid_record):
    assert pipe.validate(dict(valid_record, V1=500.0))
    assert pipe.validate(dict(valid_record, Time=-5))
    assert pipe.validate(dict(valid_record, V3="abc"))


def test_probabilities_in_unit_interval(pipe, valid_record):
    r = pipe.assess(valid_record, explain=False)
    assert 0 <= r["dnn_prob"] <= 1 and 0 <= r["lr_prob"] <= 1


def test_recommendation_in_allowed_set(pipe, valid_record):
    assert pipe.assess(valid_record, explain=False)["recommendation"] in set(C.RECOMMENDATIONS.values())


def test_report_nonempty_for_valid_and_invalid(pipe, valid_record):
    assert len(pipe.generate_report(pipe.assess(valid_record))) > 100
    rep = pipe.generate_report(pipe.assess(dict(valid_record, Amount=-1)))
    assert "TRANSACTION REJECTED" in rep


def test_trust_score_positive(pipe, valid_record):
    assert pipe.trust_score(valid_record) > 0


def test_routing_rules(pipe):
    t = pipe.threshold
    assert pipe.route(t + 0.1, 0.9, 1.5) == "AUTO-BLOCK"
    assert pipe.route(0.0, 0.0, 1.5) == "AUTO-APPROVE"
    assert pipe.route(t + 0.1, 0.0, 1.5) == "MANUAL REVIEW"
    assert pipe.route(0.0, 0.0, 0.5) == "MANUAL REVIEW"


def test_batch_of_20_runs_with_invalid_rows(pipe, test_df):
    b = test_df.head(20)[C.INPUT_COLUMNS].copy()
    b.loc[2, "Amount"] = -1
    b.loc[7, "V14"] = None
    out = pipe.score_batch(b)
    assert len(out) == 20 and (out.verdict == "TRANSACTION REJECTED").sum() == 2
    assert out.loc[2, "errors"] != ""


def test_batch_vs_single_parity(pipe, test_df):
    b = test_df.head(20)[C.INPUT_COLUMNS]
    out = pipe.score_batch(b)
    for i in range(20):
        s = pipe.assess(b.iloc[i].to_dict(), explain=False)
        assert s["verdict"] == out.loc[i, "verdict"]
        assert abs(s["dnn_prob"] - out.loc[i, "dnn_prob"]) < 1e-5
        assert abs(s["lr_prob"] - out.loc[i, "lr_prob"]) < 1e-6


def test_latency_under_criterion(pipe, valid_record):
    pipe.assess(valid_record, explain=False)
    ts = []
    for _ in range(20):
        t = time.perf_counter(); pipe.assess(valid_record, explain=False); ts.append(time.perf_counter() - t)
    assert np.percentile(ts, 95) < 0.1          # 100 ms criterion


def test_shap_top5_has_direction_and_sign(pipe, valid_record):
    for f in pipe.explain(valid_record):
        assert f["direction"] in ("increases fraud risk", "decreases fraud risk")
        assert (f["shap"] > 0) == (f["direction"] == "increases fraud risk")
