"""Central place for settings so nothing is hard-coded inside the pipeline steps."""
from pathlib import Path

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("outputs")

# The "today" of the analysis. Everything is computed relative to this date.
SNAPSHOT_DATE = "2024-12-31"

# A customer is "churned" if they make no purchase in this many days after the cutoff.
CHURN_WINDOW_DAYS = 90
# Only customers who bought something in this many days before the cutoff count as "active".
LOOKBACK_DAYS = 90
# Ignore brand-new customers - not enough history to say anything.
MIN_TENURE_DAYS = 30

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = [
    "age",
    "monthly_fee",
    "support_tickets",
    "is_promo_signup",
    "tenure_days",
    "n_orders",
    "total_spend",
    "avg_order_value",
    "recency_days",
    "orders_last_30d",
    "orders_last_90d",
    "orders_per_month",
    "activity_trend",
    "app_share",
]
CATEGORICAL_FEATURES = ["plan", "region"]
TARGET = "churned"
