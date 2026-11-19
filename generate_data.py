"""Generate a synthetic, deliberately messy dataset so the pipeline has something to run on.

Writes data/raw/customers.csv and data/raw/transactions.csv.
Swap these for real exports and the pipeline should work the same way.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_CUSTOMERS = 3000
START = pd.Timestamp("2023-01-01")
SNAPSHOT = pd.Timestamp("2024-12-31")
PLAN_FEES = {"basic": 9.99, "standard": 19.99, "premium": 39.99}
ORDERS_PER_MONTH = {"basic": 2.0, "standard": 3.0, "premium": 4.0}
OUT_DIR = Path("data/raw")


def make_customers() -> pd.DataFrame:
    n = N_CUSTOMERS
    max_offset = (SNAPSHOT - START).days - 180  # everyone has at least ~6 months of possible history
    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "signup_date": START + pd.to_timedelta(RNG.integers(0, max_offset, n), unit="D"),
            "plan": RNG.choice(list(PLAN_FEES), n, p=[0.5, 0.35, 0.15]),
            "region": RNG.choice(["North", "South", "East", "West"], n),
            "age": RNG.normal(38, 12, n).clip(18, 80).round(),
            "support_tickets": RNG.poisson(1.5, n),
            "is_promo_signup": RNG.random(n) < 0.25,
        }
    )
    df["monthly_fee"] = df["plan"].map(PLAN_FEES)

    # Hidden "truth": who churns. Driven by a few attributes so the model has signal to find.
    logit = (
        -1.0
        + 0.45 * df["support_tickets"]
        - 0.6 * (df["plan"] == "premium")
        - 0.3 * (df["plan"] == "standard")
        + 0.8 * df["is_promo_signup"]
        - 0.015 * (df["age"] - 38)
        + 0.3 * (df["region"] == "South")
    )
    df["will_churn"] = RNG.random(n) < 1 / (1 + np.exp(-logit))
    return df


def make_transactions(customers: pd.DataFrame) -> pd.DataFrame:
    ids, dates, amounts = [], [], []
    for c in customers.itertuples(index=False):
        end = SNAPSHOT
        if c.will_churn:
            lifetime = (SNAPSHOT - c.signup_date).days
            end = c.signup_date + pd.Timedelta(days=int(RNG.integers(20, lifetime - 90)))
        active_days = (end - c.signup_date).days
        n_orders = RNG.poisson(ORDERS_PER_MONTH[c.plan] * active_days / 30)
        if n_orders == 0:
            continue
        offsets = np.sort(RNG.integers(0, active_days + 1, n_orders))
        ids.append(np.full(n_orders, c.customer_id))
        dates.append((c.signup_date + pd.to_timedelta(offsets, unit="D")).values)
        amounts.append(RNG.gamma(2.0, 12 + c.monthly_fee / 2, n_orders).round(2))

    tx = pd.DataFrame(
        {
            "customer_id": np.concatenate(ids),
            "order_date": np.concatenate(dates),
            "amount": np.concatenate(amounts),
        }
    )
    tx["channel"] = RNG.choice(["web", "app"], len(tx), p=[0.6, 0.4])
    tx.insert(0, "transaction_id", [f"T{i:07d}" for i in range(len(tx))])
    return tx


def add_mess(customers: pd.DataFrame, tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject the kinds of problems real exports have."""
    customers = customers.drop(columns="will_churn").copy()
    n = len(customers)

    lower = RNG.random(n) < 0.10
    customers.loc[lower, "region"] = customers.loc[lower, "region"].str.lower() + " "
    upper = RNG.random(n) < 0.05
    customers.loc[upper, "region"] = customers.loc[upper, "region"].str.upper()
    caps = RNG.random(n) < 0.03
    customers.loc[caps, "plan"] = " " + customers.loc[caps, "plan"].str.title()
    customers.loc[RNG.random(n) < 0.03, "age"] = np.nan
    customers.loc[RNG.random(n) < 0.005, "age"] = 999
    customers = pd.concat([customers, customers.sample(20, random_state=1)])

    tx = tx.copy()
    m = len(tx)
    tx["order_date"] = tx["order_date"].dt.strftime("%Y-%m-%d")
    tx.loc[RNG.random(m) < 0.002, "order_date"] = "unknown"
    tx.loc[RNG.random(m) < 0.005, "amount"] = np.nan
    neg = RNG.random(m) < 0.005
    tx.loc[neg, "amount"] = -tx.loc[neg, "amount"]
    tx = pd.concat([tx, tx.sample(frac=0.01, random_state=1)]).sample(frac=1, random_state=2)
    return customers, tx


def main() -> None:
    customers = make_customers()
    tx = make_transactions(customers)
    customers, tx = add_mess(customers, tx)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    customers.to_csv(OUT_DIR / "customers.csv", index=False)
    tx.to_csv(OUT_DIR / "transactions.csv", index=False)
    print(f"Wrote {len(customers):,} customers and {len(tx):,} transactions to {OUT_DIR}/")


if __name__ == "__main__":
    main()
