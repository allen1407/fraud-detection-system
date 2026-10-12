"""Data loading, cleaning and the stratified 70/15/15 split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_clean() -> pd.DataFrame:
    if not C.CLEAN_CSV.exists():
        raise FileNotFoundError("Run `python scripts/prepare_data.py data/raw/<zip>` first.")
    df = pd.read_csv(C.CLEAN_CSV, parse_dates=[C.DATE_COL])
    df = df.drop_duplicates("Transaction_ID").dropna(subset=C.REQUIRED_FIELDS + [C.TARGET])
    df = df[df["Transaction_Amount"] > 0]
    return df.reset_index(drop=True)


def split_and_save(df: pd.DataFrame):
    train, rest = train_test_split(df, test_size=1 - C.SPLIT[0], stratify=df[C.TARGET], random_state=C.SEED)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest[C.TARGET], random_state=C.SEED)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.reset_index(drop=True).to_csv(C.PROCESSED_DIR / f"{name}.csv", index=False)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_splits():
    return tuple(pd.read_csv(p, parse_dates=[C.DATE_COL]) for p in (C.TRAIN_CSV, C.VAL_CSV, C.TEST_CSV))
