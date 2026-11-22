"""Load and clean the UCI Online Retail II transaction log."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_XLSX = DATA_DIR / "online_retail_II.xlsx"
CLEAN_CSV = DATA_DIR / "transactions_clean.csv"


def load_raw() -> pd.DataFrame:
    """Read both sheets (2009-10 and 2010-11) of the raw Excel file."""
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None)
    return pd.concat(sheets.values(), ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Invoice"] = df["Invoice"].astype(str)
    df = df.dropna(subset=["Customer ID"])
    df = df[~df["Invoice"].str.startswith("C")]  # cancellations
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["Price"]
    return df.drop_duplicates().reset_index(drop=True)


def get_transactions(refresh: bool = False) -> pd.DataFrame:
    """Cleaned transactions, cached as CSV after the first (slow) Excel read."""
    if CLEAN_CSV.exists() and not refresh:
        return pd.read_csv(CLEAN_CSV, parse_dates=["InvoiceDate"])
    df = clean(load_raw())
    df.to_csv(CLEAN_CSV, index=False)
    return df
