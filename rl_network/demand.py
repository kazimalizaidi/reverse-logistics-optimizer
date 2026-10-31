"""
Demand estimates: how many units of each disposition arrive at each hub, plus the
average price and weight of those units. Three ways to build one:

  from_probs   expected volumes from classifier probabilities  (the real use case)
  from_labels  actual volumes from true dispositions           (oracle / realized world)
  from_prior   hub totals x historical class mix               (no-classifier baseline)
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .config import D


@dataclass
class Demand:
    vol: np.ndarray          # H x D units
    avg_price: np.ndarray    # H x D USD per unit
    avg_weight: np.ndarray   # H x D kg per unit

    @property
    def total(self) -> float:
        return float(self.vol.sum())


def _safe_div(a, b, fallback):
    return np.where(b > 1e-12, a / np.where(b > 1e-12, b, 1), fallback)


def from_weights(df: pd.DataFrame, w: np.ndarray, H: int) -> Demand:
    """w: N x D matrix of (fractional) class membership. Probabilities -> expected
    demand; one-hot labels -> realized demand."""
    vol = np.zeros((H, D))
    pw = np.zeros((H, D))
    ww = np.zeros((H, D))
    hub = df["hub_idx"].values
    price = df["price"].values
    weight = df["weight_kg"].values
    for h in range(H):
        m = hub == h
        vol[h] = w[m].sum(axis=0)
        pw[h] = (w[m] * price[m, None]).sum(axis=0)
        ww[h] = (w[m] * weight[m, None]).sum(axis=0)
    overall_p = price.mean()
    overall_w = weight.mean()
    return Demand(vol, _safe_div(pw, vol, overall_p), _safe_div(ww, vol, overall_w))


def from_probs(df: pd.DataFrame, probs: np.ndarray, H: int) -> Demand:
    return from_weights(df, probs, H)


def from_labels(df: pd.DataFrame, H: int) -> Demand:
    return from_weights(df, np.eye(D)[df["disposition_idx"].values], H)


def from_prior(df: pd.DataFrame, prior: np.ndarray, H: int) -> Demand:
    """Each hub's actual return count split by the historical class mix; price/weight
    use the hub-wide averages (no class information)."""
    hub = df["hub_idx"].values
    vol = np.zeros((H, D))
    avg_p = np.zeros((H, D))
    avg_w = np.zeros((H, D))
    for h in range(H):
        m = hub == h
        vol[h] = m.sum() * prior
        avg_p[h] = df["price"].values[m].mean()
        avg_w[h] = df["weight_kg"].values[m].mean()
    return Demand(vol, avg_p, avg_w)
