"""The one operation: train a churn model and score current customers."""
import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE,
                        TARGET, TEST_SIZE)

logger = logging.getLogger(__name__)
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def make_pipeline() -> Pipeline:
    # Preprocessing lives inside the Pipeline so it is fit on training data only.
    prep = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    return Pipeline([("prep", prep), ("model", model)])


def train(df: pd.DataFrame) -> Pipeline:
    """Check performance on a held-out split, then refit on all labelled rows."""
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    pipe = make_pipeline().fit(X_train, y_train)
    auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
    logger.info("Test AUC: %.3f", auc)

    return make_pipeline().fit(X, y)
