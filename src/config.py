"""Central configuration: paths, seed, cost assumptions, schema."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SAMPLES_DIR = ROOT / "data" / "samples"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT / "docs"

CLEAN_CSV = PROCESSED_DIR / "transactions_clean.csv"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"

LR_PATH = MODELS_DIR / "fraud_lr_model.pkl"
DNN_PATH = MODELS_DIR / "fraud_dnn_model.keras"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METADATA_PATH = MODELS_DIR / "metadata.json"

SEED = 42
TARGET = "Class"
SPLIT = (0.70, 0.15, 0.15)

# Cost assumptions (USD)
COST_FN = 500.0   # missed fraud
COST_FP = 15.0    # legitimate transaction blocked / reviewed

LR_THRESHOLD = 0.5
# Business guardrail used when choosing the operating threshold (see docs/RESPONSIBLE_AI_REPORT.md)
MAX_VAL_FPR = 0.30
TRUST_MIN = 1.0
KNN_K = 5

# Input schema (what a caller supplies): Kaggle ULB credit-card data, V1-V28 are anonymised PCA components
TIME_COL = "Time"                                  # seconds since the first transaction in the 48 h window
PCA_INPUTS = [f"V{i}" for i in range(1, 29)]
NUMERIC_INPUTS = [TIME_COL] + PCA_INPUTS + ["Amount"]
PERTURB_INPUTS = PCA_INPUTS + ["Amount"]           # continuous inputs used for noise / robustness tests
REQUIRED_FIELDS = NUMERIC_INPUTS
INPUT_COLUMNS = REQUIRED_FIELDS

RANGES = {TIME_COL: (0.0, 172_800.0), "Amount": (0.0, 1_000_000.0), **{v: (-150.0, 150.0) for v in PCA_INPUTS}}

# Trust-score reference set: every training fraud + this many sampled legitimate rows
TRUST_REF_LEGIT = 20_000

VERDICTS = ["AUTO-BLOCK", "AUTO-APPROVE", "MANUAL REVIEW", "TRANSACTION REJECTED"]
RECOMMENDATIONS = {
    "AUTO-BLOCK": "Block the transaction and notify the customer.",
    "AUTO-APPROVE": "Approve the transaction.",
    "MANUAL REVIEW": "Escalate to a human fraud analyst before deciding.",
    "TRANSACTION REJECTED": "Correct the input errors and resubmit.",
}
