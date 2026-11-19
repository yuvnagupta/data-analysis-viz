"""Step 2: fix types, standardise text, drop rows that can't be trusted.

Every drop is logged so you can explain in the report how much data was lost and why.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _log_drop(step: str, before: int, after: int) -> None:
    if before != after:
        logger.info("  %-35s dropped %5d rows (%.1f%%)", step, before - after, 100 * (before - after) / before)


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning customers")
    df = df.copy()

    n = len(df)
    df = df.drop_duplicates(subset="customer_id", keep="first")
    _log_drop("duplicate customer_id", n, len(df))

    df["region"] = df["region"].str.strip().str.title()
    df["plan"] = df["plan"].str.strip().str.lower()

    n = len(df)
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df = df.dropna(subset=["signup_date"])
    _log_drop("unparseable signup_date", n, len(df))

    # Impossible ages -> treat as missing, then fill with the median.
    bad_age = ~df["age"].between(18, 100)
    logger.info("  %-35s %5d values set to median", "missing/invalid age", int(bad_age.sum()))
    df.loc[bad_age, "age"] = np.nan
    df["age"] = df["age"].fillna(df["age"].median())

    df["is_promo_signup"] = df["is_promo_signup"].astype(bool).astype(int)
    return df.reset_index(drop=True)


def clean_transactions(df: pd.DataFrame, valid_customer_ids: pd.Series) -> pd.DataFrame:
    logger.info("Cleaning transactions")
    df = df.copy()

    n = len(df)
    df = df.drop_duplicates()
    _log_drop("exact duplicates", n, len(df))

    n = len(df)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df = df.dropna(subset=["order_date"])
    _log_drop("unparseable order_date", n, len(df))

    n = len(df)
    df = df.dropna(subset=["amount"])
    _log_drop("missing amount", n, len(df))

    # Negative amounts are refunds. For an activity-based churn model we just ignore them.
    n = len(df)
    df = df[df["amount"] > 0]
    _log_drop("non-positive amount (refunds)", n, len(df))

    n = len(df)
    df = df[df["customer_id"].isin(valid_customer_ids)]
    _log_drop("unknown customer_id", n, len(df))

    df["channel"] = df["channel"].str.strip().str.lower()
    return df.sort_values("order_date").reset_index(drop=True)
