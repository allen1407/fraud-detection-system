import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

_results = []


@pytest.fixture(scope="session")
def pipe():
    from src.pipeline import FraudPipeline
    return FraudPipeline()


@pytest.fixture(scope="session")
def test_df():
    return pd.read_csv(C.TEST_CSV)


@pytest.fixture
def valid_record(test_df):
    return test_df.iloc[0][C.INPUT_COLUMNS].to_dict()


def pytest_runtest_logreport(report):
    if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
        _results.append(dict(test=report.nodeid.split("::")[-1], file=report.nodeid.split("::")[0],
                             result=report.outcome.upper().replace("PASSED", "PASSED"),
                             duration_s=round(report.duration, 3)))


def pytest_sessionfinish(session, exitstatus):
    if _results:
        C.REPORTS_DIR.mkdir(exist_ok=True)
        pd.DataFrame(_results).to_csv(C.REPORTS_DIR / "testing_report.csv", index=False)
