"""
Demand estimates: how many returns of each disposition reach each hub in the planning week,
with the average price and weight of those returns.

  expected_demand  classifier probabilities x assumed disposition mix   (the proposed system)
  realized_demand  the REAL return candidates of the week, with simulated disposition draws
  prior_demand     deliveries x historical hub return rate x historical mix  (no classifier)

Units are returned ORDERS (parcels). Disposition outcomes are ASSUMED (config.DISPOSITION_MIX);
what is real is WHICH orders come back, where from, and their price and weight.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import D
from .olist_data import disposition_probs, sample_dispositions


@dataclass
class Demand:
    vol: np.ndarray          # H x D expected or actual units
    avg_price: np.ndarray    # H x D  R$ per unit
    avg_weight: np.ndarray   # H x D  kg per unit

    @property
    def total(self) -> float:
        return float(self.vol.sum())


def _safe_div(a, b, fallback):
    return np.where(b > 1e-12, a / np.where(b > 1e-12, b, 1), fallback)


def from_weights(df: pd.DataFrame, w: np.ndarray, H: int, fallback_price: float, fallback_weight: float) -> Demand:
    """w: N x D matrix of (fractional) membership. Probabilities give expected demand; one-hot
    rows give realized demand."""
    vol = np.zeros((H, D)); pw = np.zeros((H, D)); ww = np.zeros((H, D))
    hub = df["hub_idx"].values
    price = df["price"].values
    weight = df["weight_kg"].values
    for h in range(H):
        m = hub == h
        vol[h] = w[m].sum(axis=0)
        pw[h] = (w[m] * price[m, None]).sum(axis=0)
        ww[h] = (w[m] * weight[m, None]).sum(axis=0)
    return Demand(vol, _safe_div(pw, vol, fallback_price), _safe_div(ww, vol, fallback_weight))


def _fallbacks(train_df: pd.DataFrame):
    r = train_df[train_df["return_candidate"] == 1]
    return float(r["price"].mean()), float(r["weight_kg"].mean())


def expected_demand(week_df: pd.DataFrame, p_return: np.ndarray, H: int, train_df: pd.DataFrame) -> Demand:
    w = p_return[:, None] * disposition_probs(week_df)
    return from_weights(week_df, w, H, *_fallbacks(train_df))


def realized_demand(week_df: pd.DataFrame, H: int, seed: int, train_df: pd.DataFrame):
    """Returns (Demand, returns_df_with_simulated_disposition_idx)."""
    ret = week_df[week_df["return_candidate"] == 1].copy()
    ret["disposition_idx"] = sample_dispositions(ret, seed)
    w = np.eye(D)[ret["disposition_idx"].values]
    return from_weights(ret, w, H, *_fallbacks(train_df)), ret


def prior_demand(train_df: pd.DataFrame, week_df: pd.DataFrame, H: int) -> Demand:
    """No-classifier baseline: this week's deliveries per hub x that hub's historical return
    rate x the historical disposition mix. Price and weight are hub averages of past returns."""
    past = train_df[train_df["return_candidate"] == 1]
    g_rate = float(train_df["return_candidate"].mean())
    g_mix = disposition_probs(past).mean(axis=0)
    fb_price, fb_weight = _fallbacks(train_df)
    vol = np.zeros((H, D)); avg_p = np.full((H, D), fb_price); avg_w = np.full((H, D), fb_weight)
    for h in range(H):
        tr_h = train_df[train_df["hub_idx"] == h]
        rate = float(tr_h["return_candidate"].mean()) if len(tr_h) else g_rate
        n_h = int((week_df["hub_idx"] == h).sum())
        vol[h] = n_h * rate * g_mix
        r_h = past[past["hub_idx"] == h]
        if len(r_h):
            avg_p[h] = r_h["price"].mean()
            avg_w[h] = r_h["weight_kg"].mean()
    return Demand(vol, avg_p, avg_w)
