"""Extract the dataset zip into data/raw/, validate columns and write data/processed/.

Usage: python scripts/prepare_data.py [path/to/archive.zip]
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

COLUMN_ROLES = {
    "Transaction_ID": "id",
    "Account_Number": "id",
    "Transaction_Date": "date",
    "Transaction_Amount": "amount",
    "Merchant_Name": "merchant",
    "Transaction_Type": "type",
    "Category": "merchant_category",
    "City": "city",
    "Country": "country",
    "Payment_Method": "payment_method",
    "Customer_Age": "age",
    "Customer_Gender": "gender",
    "Customer_Occupation": "occupation",
    "Customer_Income": "income",
    "Account_Balance": "balance",
    "Transaction_Status": "status",
    "Fraud_Flag": "label",
    "Discount_Applied": "discount",
    "Loyalty_Points_Earned": "loyalty_points",
    "Transaction_Description": "free_text",
}


def find_zip(arg):
    if arg:
        return Path(arg)
    for cand in [RAW, ROOT, ROOT.parent]:
        zips = sorted(cand.glob("*.zip"))
        if zips:
            return zips[0]
    sys.exit("No zip found. Pass the path: python scripts/prepare_data.py <zip>")


def main():
    zpath = find_zip(sys.argv[1] if len(sys.argv) > 1 else None)
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if zpath.resolve().parent != RAW.resolve():
        shutil.copy(zpath, RAW / zpath.name)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(RAW)
        print("Extracted:", z.namelist())

    csvs = sorted(RAW.glob("*.csv"))
    if not csvs:
        sys.exit("No CSV found in zip")
    main_csv = max(csvs, key=lambda p: p.stat().st_size)
    df = pd.read_csv(main_csv, parse_dates=["Transaction_Date"])

    missing = [c for c in COLUMN_ROLES if c not in df.columns]
    if missing:
        sys.exit(f"Missing expected columns: {missing}")
    df["is_fraud"] = (df["Fraud_Flag"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df = df.drop_duplicates("Transaction_ID").reset_index(drop=True)

    df.to_csv(PROCESSED / "transactions_clean.csv", index=False)
    summary = {
        "source_csv": main_csv.name,
        "rows": int(len(df)),
        "fraud_rows": int(df.is_fraud.sum()),
        "fraud_rate": round(float(df.is_fraud.mean()), 4),
        "date_min": str(df.Transaction_Date.min()),
        "date_max": str(df.Transaction_Date.max()),
        "missing_values": int(df.isna().sum().sum()),
        "column_roles": COLUMN_ROLES,
    }
    (PROCESSED / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "column_roles"}, indent=2))


if __name__ == "__main__":
    main()
