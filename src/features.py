"""RFM feature engineering and churn labelling from transactional logs.

Time split (avoids leakage):
  * features are built only from purchases BEFORE the cutoff date
  * churn label / future revenue come from the window AFTER the cutoff
A customer is churned if they made no purchase in the label window.
"""
import pandas as pd

LABEL_WINDOW_DAYS = 90

FEATURES = [
    "recency", "frequency", "monetary", "tenure",
    "avg_order_value", "avg_days_between_orders", "n_products",
]


def get_cutoff(df: pd.DataFrame, window_days: int = LABEL_WINDOW_DAYS) -> pd.Timestamp:
    return df["InvoiceDate"].max().normalize() - pd.Timedelta(days=window_days)


def build_rfm(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    hist = df[df["InvoiceDate"] < cutoff]
    orders = hist.groupby(["Customer ID", "Invoice"]).agg(
        order_date=("InvoiceDate", "max"), order_value=("Revenue", "sum")
    ).reset_index()

    g = orders.groupby("Customer ID")
    feats = pd.DataFrame({
        "recency": (cutoff - g["order_date"].max()).dt.days,          # R
        "frequency": g["Invoice"].nunique(),                           # F
        "monetary": g["order_value"].sum(),                            # M
        "tenure": (cutoff - g["order_date"].min()).dt.days,
        "avg_order_value": g["order_value"].mean(),
    })
    span = (g["order_date"].max() - g["order_date"].min()).dt.days
    feats["avg_days_between_orders"] = (span / (feats["frequency"] - 1).clip(lower=1)).where(
        feats["frequency"] > 1, feats["tenure"]
    )
    feats["n_products"] = hist.groupby("Customer ID")["StockCode"].nunique()
    return feats


def build_dataset(df: pd.DataFrame, window_days: int = LABEL_WINDOW_DAYS):
    """Return (features + churn + future_revenue per customer, cutoff)."""
    cutoff = get_cutoff(df, window_days)
    feats = build_rfm(df, cutoff)
    future = df[df["InvoiceDate"] >= cutoff].groupby("Customer ID")["Revenue"].sum()
    feats["future_revenue"] = future.reindex(feats.index).fillna(0.0)
    feats["churn"] = (feats["future_revenue"] == 0).astype(int)
    return feats, cutoff
