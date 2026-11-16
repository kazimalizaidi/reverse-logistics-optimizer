"""
Calibrate the logistics cost model on REAL Olist shipments.

Olist records, for every delivered order, the freight paid, the parcel weight, the carrier
hand-off and delivery timestamps, and the seller and customer locations. From these we fit

  freight (R$ per parcel)  = base + per_km * km + per_kg_km * kg * km
  transit (days, carrier -> customer) = base_days + coef * sqrt(km)

with MEDIAN regression (QuantileRegressor, q = 0.5): freight and delivery times have long right
tails (remote areas, strikes, outliers) that would drag an ordinary least-squares fit upward.

The reverse leg is assumed to cost the same as the outbound leg (an ASSUMPTION; Olist has no
reverse-freight data). Only data available BEFORE the planning week is used.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from .config import Econ


def calibrate_logistics(train_df: pd.DataFrame, seed: int = 0, sample: int = 10000) -> dict:
    d = train_df[(train_df.seller_customer_km > 0) & (train_df.freight_value > 0)]
    d = d.sample(min(sample, len(d)), random_state=seed)
    km = d.seller_customer_km.values
    kg = d.weight_kg.values

    q = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(np.c_[km, km * kg], d.freight_value.values)
    pred_f = q.predict(np.c_[km, km * kg])
    freight = dict(ship_base_cost=float(q.intercept_), ship_cost_per_km=float(q.coef_[0]),
                   ship_cost_per_kg_km=float(q.coef_[1]),
                   mae_r=float(np.abs(d.freight_value.values - pred_f).mean()),
                   mae_baseline_r=float(np.abs(d.freight_value.values - np.median(d.freight_value.values)).mean()))

    t = d.transit_days.values
    keep = t < np.percentile(t, 99)                       # ignore extreme outliers (lost parcels)
    qt = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(np.sqrt(km[keep]).reshape(-1, 1), t[keep])
    pred_t = qt.predict(np.sqrt(km[keep]).reshape(-1, 1))
    transit = dict(transit_base_days=float(qt.intercept_), transit_sqrt_coef=float(qt.coef_[0]),
                   mae_days=float(np.abs(t[keep] - pred_t).mean()),
                   mae_baseline_days=float(np.abs(t[keep] - np.median(t[keep])).mean()))
    return dict(n_orders=int(len(d)), freight=freight, transit=transit)


def apply_calibration(econ: Econ, cal: dict) -> Econ:
    from dataclasses import replace
    f, t = cal["freight"], cal["transit"]
    return replace(econ, ship_base_cost=f["ship_base_cost"], ship_cost_per_km=f["ship_cost_per_km"],
                   ship_cost_per_kg_km=f["ship_cost_per_kg_km"],
                   transit_base_days=t["transit_base_days"], transit_sqrt_coef=t["transit_sqrt_coef"])
