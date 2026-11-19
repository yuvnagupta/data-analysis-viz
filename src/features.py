"""Step 3: turn raw transactions into one row per customer (features + label).

Leakage note: features are built ONLY from data on or before `as_of`. The label looks
at the window AFTER the cutoff. If features peeked into that window, the model would
look great in testing and be useless in real life.
"""
import pandas as pd


def active_customer_ids(transactions: pd.DataFrame, as_of: pd.Timestamp, lookback_days: int) -> pd.Index:
    """Customers with at least one order in the `lookback_days` before `as_of`."""
    start = as_of - pd.Timedelta(days=lookback_days)
    recent = transactions[(transactions["order_date"] > start) & (transactions["order_date"] <= as_of)]
    return pd.Index(recent["customer_id"].unique())


def make_labels(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    snapshot: pd.Timestamp,
    churn_window_days: int,
    lookback_days: int,
    min_tenure_days: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Label = 1 if an active customer makes no purchase in the window after the cutoff."""
    cutoff = snapshot - pd.Timedelta(days=churn_window_days)

    active = active_customer_ids(transactions, cutoff, lookback_days)
    old_enough = customers.loc[
        customers["signup_date"] <= cutoff - pd.Timedelta(days=min_tenure_days), "customer_id"
    ]
    eligible = active.intersection(old_enough)

    future = transactions[(transactions["order_date"] > cutoff) & (transactions["order_date"] <= snapshot)]
    stayed = set(future["customer_id"])

    labels = pd.DataFrame({"customer_id": eligible})
    labels["churned"] = (~labels["customer_id"].isin(stayed)).astype(int)
    return labels, cutoff


def build_features(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    as_of: pd.Timestamp,
    customer_ids,
) -> pd.DataFrame:
    """RFM-style behaviour features + customer attributes, computed as of `as_of`."""
    hist = transactions[
        (transactions["order_date"] <= as_of) & (transactions["customer_id"].isin(customer_ids))
    ]
    g = hist.groupby("customer_id")

    feats = pd.DataFrame(
        {
            "n_orders": g.size(),
            "total_spend": g["amount"].sum(),
            "avg_order_value": g["amount"].mean(),
            "last_order": g["order_date"].max(),
            "app_share": g["channel"].apply(lambda s: (s == "app").mean()),
        }
    )
    feats["recency_days"] = (as_of - feats["last_order"]).dt.days

    for days in (30, 90):
        window = hist[hist["order_date"] > as_of - pd.Timedelta(days=days)]
        feats[f"orders_last_{days}d"] = window.groupby("customer_id").size().reindex(feats.index, fill_value=0)

    df = customers[customers["customer_id"].isin(customer_ids)].merge(
        feats, left_on="customer_id", right_index=True, how="left"
    )

    df["tenure_days"] = (as_of - df["signup_date"]).dt.days
    df["orders_per_month"] = df["n_orders"] / (df["tenure_days"] / 30).clip(lower=1)
    # >1 means the last month was busier than the 3-month average, <1 means slowing down.
    df["activity_trend"] = (df["orders_last_30d"] * 3) / df["orders_last_90d"].clip(lower=1)

    fill_zero = ["n_orders", "total_spend", "avg_order_value", "orders_last_30d",
                 "orders_last_90d", "app_share", "orders_per_month", "activity_trend"]
    df[fill_zero] = df[fill_zero].fillna(0)
    df["recency_days"] = df["recency_days"].fillna(df["tenure_days"])

    return df.drop(columns=["last_order", "signup_date"]).reset_index(drop=True)
