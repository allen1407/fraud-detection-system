"""Copy the dataset (CSV or zip) into data/raw/, validate columns and write data/processed/.

Usage: python scripts/prepare_data.py [path/to/creditcard.csv | path/to/archive.zip]
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
    "Time": "seconds since the first transaction (48 h window)",
    **{f"V{i}": "anonymised PCA component" for i in range(1, 29)},
    "Amount": "amount",
    "Class": "label (1 = fraud)",
}


def find_source(arg):
    if arg:
        return Path(arg)
    for cand in [RAW, ROOT, ROOT.parent]:
        for pattern in ("creditcard*.csv", "*.zip"):
            hits = sorted(cand.glob(pattern))
            if hits:
                return hits[0]
    sys.exit("No dataset found. Pass the path: python scripts/prepare_data.py <creditcard.csv>")


def main():
    src = find_source(sys.argv[1] if len(sys.argv) > 1 else None)
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if src.resolve().parent != RAW.resolve():
        shutil.copy(src, RAW / src.name)
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            z.extractall(RAW)
            print("Extracted:", z.namelist())
        main_csv = max(RAW.glob("*.csv"), key=lambda p: p.stat().st_size)
    else:
        main_csv = RAW / src.name
    df = pd.read_csv(main_csv)

    missing = [c for c in COLUMN_ROLES if c not in df.columns]
    if missing:
        sys.exit(f"Missing expected columns: {missing}")
    n_raw = len(df)
    df["Class"] = df["Class"].astype(int)
    df = df.drop_duplicates().reset_index(drop=True)

    df.to_csv(PROCESSED / "transactions_clean.csv", index=False)
    summary = {
        "source_csv": main_csv.name,
        "raw_rows": int(n_raw),
        "duplicates_removed": int(n_raw - len(df)),
        "rows": int(len(df)),
        "fraud_rows": int(df.Class.sum()),
        "fraud_rate": round(float(df.Class.mean()), 5),
        "time_span_hours": round(float(df.Time.max() - df.Time.min()) / 3600, 1),
        "amount_max": float(df.Amount.max()),
        "missing_values": int(df.isna().sum().sum()),
        "column_roles": COLUMN_ROLES,
    }
    (PROCESSED / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "column_roles"}, indent=2))


if __name__ == "__main__":
    main()
