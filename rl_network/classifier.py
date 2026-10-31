"""
Disposition classifier: predicts restock / refurbish / scrap for a return BEFORE it
reaches a warehouse, using only information available at pickup.

Model: soft-voting ensemble (RandomForest + HistGradientBoosting + LogisticRegression)
wrapped in isotonic probability calibration. Calibration matters here because the
optimizer consumes the probabilities as expected volumes, not just the argmax label.
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DISPOSITIONS

NUM_FEATURES = ["price", "weight_kg", "customer_return_rate", "customer_orders_12m",
                "customer_tenure_months", "days_since_purchase", "transit_days_to_hub"]
CAT_FEATURES = ["category", "return_reason", "reported_condition", "origin_hub"]
FEATURES = NUM_FEATURES + CAT_FEATURES


def _ensemble(seed: int) -> Pipeline:
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, n_jobs=-1, random_state=seed)
    hgb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250,
                                         random_state=seed)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    vote = VotingClassifier([("rf", rf), ("hgb", hgb), ("lr", lr)], voting="soft",
                            weights=[1, 2, 1])
    return Pipeline([("pre", pre), ("clf", vote)])


def train(train_df: pd.DataFrame, seed: int):
    """Returns (raw_model, calibrated_model)."""
    X, y = train_df[FEATURES], train_df["disposition_idx"].values
    raw = _ensemble(seed).fit(X, y)
    cal = CalibratedClassifierCV(_ensemble(seed), method="isotonic", cv=3).fit(X, y)
    return raw, cal


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Top-label ECE: weighted mean |accuracy - confidence| over confidence bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def multiclass_brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def evaluate(probs: np.ndarray, y: np.ndarray, true_probs: np.ndarray, prior: np.ndarray) -> dict:
    """Metrics vs. three reference points: the majority-class guess, the training-prior
    probabilities (no-model), and the Bayes-optimal oracle (generator's true probabilities)."""
    pred = probs.argmax(axis=1)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "brier": multiclass_brier(probs, y),
        "ece_top_label": expected_calibration_error(probs, y),
        "confusion_matrix(rows=true, cols=pred)": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
        "classes": DISPOSITIONS,
        "reference_majority_class_accuracy": float((y == prior.argmax()).mean()),
        "reference_prior_log_loss": float(log_loss(y, np.tile(prior, (len(y), 1)), labels=[0, 1, 2])),
        "reference_oracle_accuracy": float(accuracy_score(y, true_probs.argmax(axis=1))),
        "reference_oracle_log_loss": float(log_loss(y, true_probs, labels=[0, 1, 2])),
    }
    return out


def feature_importance(raw_model: Pipeline, df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permutation importance of ORIGINAL (pre-encoding) features, measured as the
    increase in log-loss when a feature is shuffled."""
    from sklearn.inspection import permutation_importance
    sub = df.sample(min(3000, len(df)), random_state=seed)
    res = permutation_importance(raw_model, sub[FEATURES], sub["disposition_idx"].values,
                                 n_repeats=5, random_state=seed, scoring="neg_log_loss", n_jobs=1)
    return (pd.DataFrame({"feature": FEATURES, "logloss_increase": res.importances_mean,
                          "std": res.importances_std})
            .sort_values("logloss_increase", ascending=False).reset_index(drop=True))
