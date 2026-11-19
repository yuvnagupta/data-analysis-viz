import pandas as pd

from src.features import build_features, make_labels

SNAPSHOT = pd.Timestamp("2024-12-31")
CUTOFF = SNAPSHOT - pd.Timedelta(days=90)


def _data():
    customers = pd.DataFrame(
        {
            "customer_id": ["A", "B", "C"],
            "signup_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "plan": ["basic"] * 3, "region": ["North"] * 3, "age": [30.0] * 3,
            "support_tickets": [0, 1, 2], "is_promo_signup": [0, 0, 1], "monthly_fee": [9.99] * 3,
        }
    )
    tx = pd.DataFrame(
        {
            "customer_id": ["A", "A", "B", "C"],
            "order_date": pd.to_datetime(["2024-09-15", "2024-11-20", "2024-09-20", "2024-03-01"]),
            "amount": [10.0, 20.0, 15.0, 5.0],
            "channel": ["app", "web", "web", "app"],
        }
    )
    return customers, tx


def test_labels():
    customers, tx = _data()
    labels, cutoff = make_labels(customers, tx, SNAPSHOT, 90, 90, 30)
    labels = labels.set_index("customer_id")["churned"]
    assert cutoff == CUTOFF
    assert labels["A"] == 0          # bought after the cutoff -> stayed
    assert labels["B"] == 1          # active before, silent after -> churned
    assert "C" not in labels.index   # not active before the cutoff -> excluded


def test_features_do_not_leak_future():
    customers, tx = _data()
    feats = build_features(customers, tx, CUTOFF, ["A"]).set_index("customer_id")
    # A's Nov 20 order is after the cutoff and must not be counted.
    assert feats.loc["A", "n_orders"] == 1
    assert feats.loc["A", "total_spend"] == 10.0
