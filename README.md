# Customer churn pipeline

Predicts which active customers will stop buying in the next 90 days.

```
data/raw/*.csv
   ├─ 1. load      check required columns exist
   ├─ 2. clean     dedupe, fix types/text, drop bad rows, impute age
   ├─ 3. features  one row per customer (recency, frequency, spend, tenure, ...)
   └─ 4. predict   logistic regression -> outputs/at_risk_customers.csv
```

## Run

```bash
pip install -r requirements.txt
python generate_data.py      # synthetic messy data; skip if you have real exports
python run_pipeline.py
pytest
```

## Key idea: no peeking at the future
The model trains on features built from data up to a cutoff (snapshot − 90 days). The
label is whether the customer bought anything in the 90 days after it. The trained model is
then applied to features as of the snapshot to predict the next 90 days.
`tests/test_features.py` checks this.
