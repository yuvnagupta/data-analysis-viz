"""Step 1: read raw files and fail early if they don't look right."""
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CUSTOMER_COLS = {"customer_id", "signup_date", "plan", "region", "age",
                 "support_tickets", "is_promo_signup", "monthly_fee"}
TRANSACTION_COLS = {"transaction_id", "customer_id", "order_date", "amount", "channel"}


def _check_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def load_raw(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    customers = pd.read_csv(data_dir / "customers.csv")
    transactions = pd.read_csv(data_dir / "transactions.csv")

    _check_columns(customers, CUSTOMER_COLS, "customers.csv")
    _check_columns(transactions, TRANSACTION_COLS, "transactions.csv")

    logger.info("Loaded %d customer rows and %d transaction rows", len(customers), len(transactions))
    return customers, transactions
