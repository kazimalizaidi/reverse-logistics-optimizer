"""
Return-propensity classifier: for each delivered order, the probability that it becomes a return
candidate (a 1-2 star review, see olist_data). This is a REAL supervised target: the label comes
from the data, so every metric below is a genuine out-of-time measurement, with no "oracle".

Features are limited to what is known when the parcel reaches the customer: price, freight,
weight, product group, seller and customer state, distance, and the delivery timeline (promised
vs actual delivery). The review score itself is never a feature.

Model: soft-voting ensemble (random forest + gradient boosting + logistic regression) with
isotonic probability calibration on the most recent slice of the history. Calibration matters because the optimizer consumes the
probabilities as EXPECTED return volumes.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_FEATURES = ["price", "freight_value", "n_items", "weight_kg", "volume_cm3", "freight_ratio",
                "delivery_days", "est_days", "delay_days", "handoff_days", "transit_days",
                "seller_customer_km", "same_state", "purchase_dow"]
CAT_FEATURES = ["category_group", "customer_state", "seller_state"]
FEATURES = NUM_FEATURES + CAT_FEATURES
TARGET = "return_candidate"


def _pipeline(seed: int, fast: bool) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])
    hgb = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=250, random_state=seed)
    if fast:                     # used for the many retrains of the rolling backtest
        return Pipeline([("pre", pre), ("clf", hgb)])
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=20, n_jobs=-1, random_state=seed)
    lr = LogisticRegression(max_iter=2000, C=1.0)
    vote = VotingClassifier([("rf", rf), ("hgb", hgb), ("lr", lr)], voting="soft", weights=[1, 2, 1])
    return Pipeline([("pre", pre), ("clf", vote)])


class IsotonicCalibrated:
    """A fitted model plus an isotonic map from its probabilities to observed frequencies."""

    def __init__(self, base, iso):
        self.base, self.iso = base, iso

    def predict_proba(self, X):
        q = self.iso.predict(self.base.predict_proba(X)[:, 1])
        return np.c_[1 - q, q]


def train(train_df: pd.DataFrame, seed: int, fast: bool = False, calib_frac: float = 0.15):
    """Fit on the older part of the history and calibrate on the NEWEST `calib_frac` of it
    (rows are ordered by delivery date), mimicking deployment: the calibration data is always
    more recent than the data the model learned from. Returns (raw_model, calibrated_model).
    `fast=True` uses a single gradient-boosting model (rolling backtest)."""
    d = train_df.sort_values("order_delivered_customer_date")
    n_cal = max(200, int(len(d) * calib_frac))
    fit_df, cal_df = d.iloc[:-n_cal], d.iloc[-n_cal:]
    raw = _pipeline(seed, fast).fit(fit_df[FEATURES], fit_df[TARGET].values)
    p_cal = raw.predict_proba(cal_df[FEATURES])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_cal, cal_df[TARGET].values)
    return raw, IsotonicCalibrated(raw, iso)


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES])[:, 1]


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Weighted mean |observed rate - mean predicted probability| over equal-count bins."""
    order = np.argsort(p)
    ece = 0.0
    for idx in np.array_split(order, bins):
        if len(idx):
            ece += len(idx) / len(p) * abs(y[idx].mean() - p[idx].mean())
    return float(ece)


def _scores(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "ece": expected_calibration_error(p, y),
            "mean_predicted": float(p.mean()), "observed_rate": float(y.mean())}


def baselines(train_df: pd.DataFrame, week_df: pd.DataFrame) -> dict:
    """Simple forecasts a planner could make WITHOUT machine learning, fitted on training data."""
    g = train_df[TARGET].mean()
    hub = train_df.groupby("hub")[TARGET].mean()
    hub_late = train_df.groupby(["hub", "late"])[TARGET].mean()
    p_global = np.full(len(week_df), g)
    p_hub = week_df["hub"].map(hub).fillna(g).values
    keys = list(zip(week_df["hub"], week_df["late"]))
    p_rule = np.array([hub_late.get(k, g) for k in keys])
    return {"global_rate": p_global, "hub_rate": p_hub, "late_x_hub_rule": p_rule}


def evaluate(p: np.ndarray, week_df: pd.DataFrame, train_df: pd.DataFrame) -> dict:
    y = week_df[TARGET].values
    out = {"model": _scores(y, p)}
    for name, pb in baselines(train_df, week_df).items():
        out[name] = _scores(y, pb)
    out["lift_top_decile"] = float(y[np.argsort(-p)[: max(1, len(p) // 10)]].mean() / max(y.mean(), 1e-9))
    return out


def calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 10) -> pd.DataFrame:
    order = np.argsort(p)
    rows = []
    for i, idx in enumerate(np.array_split(order, bins), 1):
        rows.append(dict(bin=i, n=len(idx), mean_predicted=float(p[idx].mean()), observed_rate=float(y[idx].mean())))
    return pd.DataFrame(rows)


def feature_importance(raw_model: Pipeline, week_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permutation importance of ORIGINAL features: drop in ROC-AUC when a feature is shuffled."""
    from sklearn.inspection import permutation_importance
    res = permutation_importance(raw_model, week_df[FEATURES], week_df[TARGET].values, n_repeats=5,
                                 random_state=seed, scoring="roc_auc", n_jobs=1)
    return (pd.DataFrame({"feature": FEATURES, "auc_drop": res.importances_mean, "std": res.importances_std})
            .sort_values("auc_drop", ascending=False).reset_index(drop=True))
