"""
Synthetic returns generator.

IMPORTANT (label circularity): disposition labels are SAMPLED from a softmax over
hand-written logits, not assigned by deterministic rules. Two returns with identical
features can therefore get different outcomes, so a classifier cannot reach 100%.
The generator also stores the true class probabilities (columns prefixed `_true_p_`),
which gives us a Bayes-optimal ceiling to compare the trained model against.
These `_true_p_` columns must NEVER be used as model features.

Replace this module with a loader for real returns data when you have it; the rest of
the pipeline only needs the columns listed in classifier.NUM_FEATURES / CAT_FEATURES
plus `price`, `weight_kg`, `hub_idx` and `disposition`.
"""
import numpy as np
import pandas as pd
from .config import HUBS, CATEGORIES, REASONS, CONDITIONS, DISPOSITIONS

# category: (median price, log-sigma, mean weight kg, reason probabilities in REASONS order)
CAT_PARAMS = {
    "electronics": (180, 0.80, 1.2, [.20, .00, .10, .15, .42, .13]),
    "appliances":  (320, 0.60, 9.0, [.12, .00, .08, .12, .50, .18]),
    "home":        (60,  0.70, 3.5, [.25, .00, .12, .18, .28, .17]),
    "apparel":     (28,  0.50, 0.5, [.22, .50, .08, .12, .05, .03]),
    "beauty":      (18,  0.50, 0.3, [.40, .00, .12, .20, .20, .08]),
    "toys":        (25,  0.60, 0.9, [.30, .00, .15, .15, .28, .12]),
}
CATEGORY_MIX = [0.22, 0.06, 0.14, 0.30, 0.14, 0.14]   # in CATEGORIES order

# P(condition | reason), order: intact, opened, damaged
COND_GIVEN_REASON = {
    "changed_mind":       [.40, .55, .05],
    "size_fit":           [.10, .80, .10],
    "wrong_item":         [.55, .40, .05],
    "not_as_described":   [.15, .70, .15],
    "defective":          [.10, .55, .35],
    "damaged_in_transit": [.03, .12, .85],
}

# Logit contributions, columns = [restock, refurbish, scrap]
REASON_EFFECT = {
    "changed_mind":       [1.8, -0.5, -1.5],
    "size_fit":           [1.5, -0.3, -1.2],
    "wrong_item":         [1.6, -0.5, -1.2],
    "not_as_described":   [0.6,  0.3, -0.4],
    "defective":          [-2.0, 1.6,  0.6],
    "damaged_in_transit": [-1.5, 0.9,  0.8],
}
CONDITION_EFFECT = {
    "intact":  [0.8, -0.2, -0.8],
    "opened":  [0.0,  0.2, -0.1],
    "damaged": [-1.8, 0.5,  1.4],
}
CATEGORY_EFFECT = {
    "electronics": [0.0, 0.6, 0.0],
    "appliances":  [-0.2, 0.5, 0.0],
    "home":        [0.0, 0.0, 0.1],
    "apparel":     [0.4, -0.4, 0.2],
    "beauty":      [0.1, -0.6, 0.4],
    "toys":        [0.2, 0.0, 0.0],
}
CLASS_BIAS = [0.15, -0.35, -0.55]


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# Regional category tilt (multiplies the base category mix, then renormalised per hub)
HUB_TILT = {
    "Karachi":    {"electronics": 1.25},
    "Lahore":     {"apparel": 1.10, "electronics": 1.10},
    "Faisalabad": {"apparel": 1.40},
    "Peshawar":   {"appliances": 1.50, "home": 1.20},
    "Quetta":     {"appliances": 1.60, "home": 1.30, "electronics": 0.80},
    "Sialkot":    {"apparel": 1.30, "toys": 1.30},
}


def generate_returns(n: int, seed: int, category_mix=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hub_names = [h[0] for h in HUBS]
    hub_share = np.array([h[3] for h in HUBS])
    hub_remote = np.array([h[4] for h in HUBS])

    hub_idx = rng.choice(len(HUBS), size=n, p=hub_share / hub_share.sum())
    base_mix = np.array(CATEGORY_MIX if category_mix is None else category_mix, dtype=float)
    category = np.empty(n, dtype=object)
    for hi, hname in enumerate(hub_names):
        m = hub_idx == hi
        tilt = np.array([HUB_TILT.get(hname, {}).get(c, 1.0) for c in CATEGORIES])
        p = base_mix * tilt
        category[m] = rng.choice(CATEGORIES, size=m.sum(), p=p / p.sum())

    price = np.empty(n)
    weight = np.empty(n)
    reason = np.empty(n, dtype=object)
    for c in CATEGORIES:
        m = category == c
        k = m.sum()
        med, sig, w, rp = CAT_PARAMS[c]
        price[m] = np.clip(rng.lognormal(np.log(med), sig, k), 3, 2500)
        weight[m] = w * rng.lognormal(0, 0.25, k)
        reason[m] = rng.choice(REASONS, size=k, p=rp)

    condition = np.empty(n, dtype=object)
    for r in REASONS:
        m = reason == r
        condition[m] = rng.choice(CONDITIONS, size=m.sum(), p=COND_GIVEN_REASON[r])

    return_rate = rng.beta(1.5, 12, n)
    orders_12m = rng.poisson(6, n) + 1
    tenure = np.clip(rng.exponential(20, n), 1, 120)
    days_since_purchase = np.clip(np.round(rng.gamma(2.2, 6.0, n)), 1, 60)
    transit_days = np.round(hub_remote[hub_idx] + rng.gamma(2.0, 0.7, n), 2)

    # ---- latent disposition probabilities (hidden from the model) ----
    logits = np.tile(CLASS_BIAS, (n, 1)).astype(float)
    logits += np.array([REASON_EFFECT[r] for r in reason])
    logits += np.array([CONDITION_EFFECT[c] for c in condition])
    logits += np.array([CATEGORY_EFFECT[c] for c in category])
    price_z = (np.log(price) - np.log(60)) / 1.0
    logits[:, 1] += 0.45 * price_z                    # pricey items worth repairing
    logits[:, 2] -= 0.55 * price_z
    dsp_z = (days_since_purchase - 13) / 8
    logits[:, 0] -= 0.30 * dsp_z                      # older purchases less resellable
    logits[:, 2] += 0.30 * dsp_z
    logits[:, 2] += 0.20 * (transit_days - 1.5)       # long transit -> more wear
    logits[:, 2] += 2.0 * (return_rate - 0.11)        # high-return customers (wear-and-return)
    logits[:, 0] -= 1.5 * (return_rate - 0.11)
    probs = _softmax(logits)

    u = rng.random(n)[:, None]
    label_idx = (u > np.cumsum(probs, axis=1)).sum(axis=1).clip(0, 2)

    df = pd.DataFrame({
        "return_id": np.arange(n),
        "hub_idx": hub_idx,
        "origin_hub": [hub_names[i] for i in hub_idx],
        "category": category,
        "price": np.round(price, 2),
        "weight_kg": np.round(weight, 3),
        "customer_return_rate": np.round(return_rate, 4),
        "customer_orders_12m": orders_12m,
        "customer_tenure_months": np.round(tenure, 1),
        "days_since_purchase": days_since_purchase.astype(int),
        "transit_days_to_hub": transit_days,
        "return_reason": reason,
        "reported_condition": condition,
        "disposition": [DISPOSITIONS[i] for i in label_idx],
        "disposition_idx": label_idx,
    })
    for j, d in enumerate(DISPOSITIONS):
        df[f"_true_p_{d}"] = probs[:, j]
    return df
