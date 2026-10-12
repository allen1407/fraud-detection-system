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
TARGET = "is_fraud"
SPLIT = (0.70, 0.15, 0.15)

# Cost assumptions (USD)
COST_FN = 500.0   # missed fraud
COST_FP = 15.0    # legitimate transaction blocked / reviewed

LR_THRESHOLD = 0.5
# Business guardrail used when choosing the operating threshold (see docs/RESPONSIBLE_AI_REPORT.md)
MAX_VAL_FPR = 0.30
TRUST_MIN = 1.0
KNN_K = 5

# Input schema (what a caller supplies)
DATE_COL = "Transaction_Date"
NUMERIC_INPUTS = ["Transaction_Amount", "Customer_Age", "Customer_Income", "Account_Balance"]
CATEGORICAL_INPUTS = ["Transaction_Type", "Category", "Payment_Method", "Customer_Gender"]
HIGH_CARD_INPUTS = ["Customer_Occupation", "Merchant_Name"]
REQUIRED_FIELDS = [DATE_COL] + NUMERIC_INPUTS + CATEGORICAL_INPUTS + HIGH_CARD_INPUTS
INPUT_COLUMNS = REQUIRED_FIELDS

RANGES = {
    "Transaction_Amount": (0.01, 1_000_000.0),
    "Customer_Age": (18, 100),
    "Customer_Income": (0.0, 10_000_000.0),
    "Account_Balance": (0.0, 100_000_000.0),
}

VERDICTS = ["AUTO-BLOCK", "AUTO-APPROVE", "MANUAL REVIEW", "TRANSACTION REJECTED"]
RECOMMENDATIONS = {
    "AUTO-BLOCK": "Block the transaction and notify the customer.",
    "AUTO-APPROVE": "Approve the transaction.",
    "MANUAL REVIEW": "Escalate to a human fraud analyst before deciding.",
    "TRANSACTION REJECTED": "Correct the input errors and resubmit.",
}
