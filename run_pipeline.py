"""Churn pipeline: raw CSVs -> clean -> features -> predict churn -> at-risk customer list.

Usage:
    python generate_data.py          # only if you don't have real data
    python run_pipeline.py
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src import clean, config, features, load, model

logger = logging.getLogger("pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    p.add_argument("--snapshot-date", default=config.SNAPSHOT_DATE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    snapshot = pd.Timestamp(args.snapshot_date)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    customers_raw, tx_raw = load.load_raw(args.data_dir)

    # 2. Clean
    customers = clean.clean_customers(customers_raw)
    tx = clean.clean_transactions(tx_raw, customers["customer_id"])

    # 3. Build a training table as of the cutoff (label = what happened after it)
    labels, cutoff = features.make_labels(
        customers, tx, snapshot,
        churn_window_days=config.CHURN_WINDOW_DAYS,
        lookback_days=config.LOOKBACK_DAYS,
        min_tenure_days=config.MIN_TENURE_DAYS,
    )
    train_df = features.build_features(customers, tx, cutoff, labels["customer_id"]).merge(labels, on="customer_id")
    logger.info("Training on %d customers, churn rate %.1f%%", len(train_df), 100 * train_df[config.TARGET].mean())

    # 4. The operation: predict churn for today's active customers
    churn_model = model.train(train_df)
    current_ids = features.active_customer_ids(tx, snapshot, config.LOOKBACK_DAYS)
    current = features.build_features(customers, tx, snapshot, current_ids)
    current["churn_probability"] = churn_model.predict_proba(current[model.FEATURES])[:, 1]

    out = args.output_dir / "at_risk_customers.csv"
    (current[["customer_id", "plan", "region", "recency_days", "churn_probability"]]
        .sort_values("churn_probability", ascending=False)
        .to_csv(out, index=False))
    logger.info("Scored %d customers -> %s", len(current), out)


if __name__ == "__main__":
    main()
