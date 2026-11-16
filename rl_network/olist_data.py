"""
Olist data layer: raw Kaggle CSVs -> one clean ORDER-level table.

Source: "Brazilian E-Commerce Public Dataset by Olist" (Kaggle, olistbr/brazilian-ecommerce),
about 100k real orders placed 2016-2018. The raw files are NOT stored in this repository
(size and licence: CC BY-NC-SA 4.0). Download them and place the CSVs in  data/olist/ .

What is REAL here: orders, prices, freight, product weights and dimensions, seller and
customer locations, delivery timestamps and review scores.

What is a PROXY: Olist has no returns table. A "return candidate" is a delivered order whose
customer left a 1 or 2 star review (config.RETURN_SCORE_MAX). This is an upper-bound stand-in
for a return, not an observed return.

What is ASSUMED (config.py): which disposition (restock / refurbish / scrap) a return ends in,
and all facility, cost and capacity figures.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (CATEGORY_GROUP_KEYWORDS, DISPOSITIONS, DISPOSITION_MIX, LATE_RESTOCK_SHIFT,
                     RETURN_SCORE_MAX, STATE_TO_HUB, HUBS)
from .network import haversine_km

REQUIRED = ["olist_orders_dataset.csv", "olist_order_items_dataset.csv",
            "olist_order_reviews_dataset.csv", "olist_customers_dataset.csv",
            "olist_products_dataset.csv", "olist_sellers_dataset.csv",
            "olist_geolocation_dataset.csv", "product_category_name_translation.csv"]

BR_LAT, BR_LON = (-34.0, 6.0), (-74.5, -33.0)      # sanity box for coordinates


def load_raw(data_dir: Path) -> dict:
    data_dir = Path(data_dir)
    missing = [f for f in REQUIRED if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Olist files missing in {data_dir}: {missing}\n"
            "Download the dataset from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce "
            "and put the CSV files in that folder.")
    date_cols = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
                 "order_delivered_customer_date", "order_estimated_delivery_date"]
    raw = {
        "orders": pd.read_csv(data_dir / "olist_orders_dataset.csv", parse_dates=date_cols),
        "items": pd.read_csv(data_dir / "olist_order_items_dataset.csv"),
        "reviews": pd.read_csv(data_dir / "olist_order_reviews_dataset.csv",
                               parse_dates=["review_creation_date", "review_answer_timestamp"]),
        "customers": pd.read_csv(data_dir / "olist_customers_dataset.csv"),
        "products": pd.read_csv(data_dir / "olist_products_dataset.csv"),
        "sellers": pd.read_csv(data_dir / "olist_sellers_dataset.csv"),
        "geo": pd.read_csv(data_dir / "olist_geolocation_dataset.csv"),
        "translation": pd.read_csv(data_dir / "product_category_name_translation.csv"),
    }
    return raw


def category_group(english_name: str) -> str:
    """Coarse product group used for the disposition assumptions."""
    if not isinstance(english_name, str):
        return "other"
    n = english_name.lower()
    for group, keys in CATEGORY_GROUP_KEYWORDS.items():
        if any(k in n for k in keys):
            return group
    return "other"


def zip_centroids(geo: pd.DataFrame) -> pd.DataFrame:
    g = geo[geo.geolocation_lat.between(*BR_LAT) & geo.geolocation_lng.between(*BR_LON)]
    c = (g.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
         .median().rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lon"}))
    return c


def build_order_table(raw: dict, road_factor: float = 1.3) -> pd.DataFrame:
    """One row per DELIVERED, REVIEWED order with features known at delivery time."""
    o = raw["orders"]
    o = o[(o.order_status == "delivered") & o.order_delivered_customer_date.notna()
          & o.order_delivered_carrier_date.notna() & o.order_estimated_delivery_date.notna()].copy()

    # ---- items -> per-order aggregates (units = one returned ORDER / parcel) ----
    prod = raw["products"].merge(raw["translation"], on="product_category_name", how="left")
    prod["category"] = prod["product_category_name_english"].fillna("unknown")
    prod["weight_kg"] = prod["product_weight_g"].where(prod["product_weight_g"] > 0) / 1000.0
    prod["volume_cm3"] = (prod["product_length_cm"] * prod["product_height_cm"] * prod["product_width_cm"])
    # impute missing weight / volume with the category median, then the global median
    for col in ("weight_kg", "volume_cm3"):
        prod[col] = prod[col].fillna(prod.groupby("category")[col].transform("median")).fillna(prod[col].median())
    it = raw["items"].merge(prod[["product_id", "category", "weight_kg", "volume_cm3"]], on="product_id", how="left")
    it["weight_kg"] = it["weight_kg"].fillna(prod["weight_kg"].median())
    it["volume_cm3"] = it["volume_cm3"].fillna(prod["volume_cm3"].median())
    it["category"] = it["category"].fillna("unknown")
    top = it.sort_values("price", ascending=False).drop_duplicates("order_id")[["order_id", "category", "seller_id"]]
    agg = it.groupby("order_id").agg(price=("price", "sum"), freight=("freight_value", "sum"),
                                     n_items=("order_item_id", "count"), weight_kg=("weight_kg", "sum"),
                                     volume_cm3=("volume_cm3", "sum"))
    agg = agg.join(top.set_index("order_id"))
    o = o.merge(agg, left_on="order_id", right_index=True, how="inner")

    # ---- locations ----
    cen = zip_centroids(raw["geo"])
    cu = raw["customers"][["customer_id", "customer_state", "customer_zip_code_prefix"]]
    se = raw["sellers"][["seller_id", "seller_state", "seller_zip_code_prefix"]]
    o = o.merge(cu, on="customer_id", how="left").merge(se, on="seller_id", how="left")
    o = (o.merge(cen.rename(columns={"lat": "cust_lat", "lon": "cust_lon"}),
                 left_on="customer_zip_code_prefix", right_index=True, how="left")
          .merge(cen.rename(columns={"lat": "sell_lat", "lon": "sell_lon"}),
                 left_on="seller_zip_code_prefix", right_index=True, how="left"))
    o = o.dropna(subset=["cust_lat", "sell_lat", "customer_state"])
    o["seller_customer_km"] = haversine_km(o.sell_lat, o.sell_lon, o.cust_lat, o.cust_lon) * road_factor
    o["same_state"] = (o.customer_state == o.seller_state).astype(int)

    # ---- timing features (all known when the parcel reaches the customer) ----
    day = pd.Timedelta(days=1)
    o["delivery_days"] = (o.order_delivered_customer_date - o.order_purchase_timestamp) / day
    o["est_days"] = (o.order_estimated_delivery_date - o.order_purchase_timestamp) / day
    o["delay_days"] = (o.order_delivered_customer_date - o.order_estimated_delivery_date) / day
    o["handoff_days"] = (o.order_delivered_carrier_date - o.order_purchase_timestamp) / day
    o["transit_days"] = (o.order_delivered_customer_date - o.order_delivered_carrier_date) / day
    o["late"] = (o.delay_days > 0).astype(int)
    o["freight_ratio"] = o.freight / o.price.clip(lower=1)
    o["purchase_dow"] = o.order_purchase_timestamp.dt.dayofweek
    o = o[(o.handoff_days >= 0) & (o.transit_days >= 0)]     # drop timestamp errors

    # ---- label: latest review per order; 1-2 stars = return candidate (PROXY) ----
    rv = (raw["reviews"].sort_values("review_answer_timestamp")
          .drop_duplicates("order_id", keep="last")[["order_id", "review_score", "review_creation_date",
                                                      "review_answer_timestamp"]])
    o = o.merge(rv, on="order_id", how="inner")
    o["return_candidate"] = (o.review_score <= RETURN_SCORE_MAX).astype(int)

    # ---- week of delivery = the week the (potential) return enters the network ----
    d = o.order_delivered_customer_date.dt.normalize()
    o["week_start"] = d - pd.to_timedelta(d.dt.dayofweek, unit="D")
    o["category_group"] = o.category.map(category_group)

    # ---- hubs ----
    hub_names = [h[0] for h in HUBS]
    o["hub"] = o.customer_state.map(STATE_TO_HUB)
    o = o.dropna(subset=["hub"])
    o["hub_idx"] = o.hub.map({h: i for i, h in enumerate(hub_names)}).astype(int)
    o = o.rename(columns={"price": "price", "freight": "freight_value"})
    keep = ["order_id", "order_purchase_timestamp", "order_delivered_customer_date", "review_creation_date",
            "review_answer_timestamp", "week_start", "hub", "hub_idx", "customer_state", "seller_state", "category", "category_group",
            "price", "freight_value", "weight_kg", "volume_cm3", "n_items", "freight_ratio",
            "delivery_days", "est_days", "delay_days", "handoff_days", "transit_days", "late",
            "seller_customer_km", "same_state", "purchase_dow", "review_score", "return_candidate"]
    return o[keep].sort_values("order_delivered_customer_date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Disposition ASSUMPTION layer (no public data has disposition outcomes)
# ---------------------------------------------------------------------------
def disposition_probs(df: pd.DataFrame) -> np.ndarray:
    """N x 3 probabilities [restock, refurbish, scrap] for a return, from product group, with
    a shift toward restock when the original delivery was late (customer rejected the delay, not
    the item). These are ASSUMPTIONS documented in config.DISPOSITION_MIX."""
    base = np.array([DISPOSITION_MIX.get(g, DISPOSITION_MIX["other"]) for g in df["category_group"]], dtype=float)
    late = df["late"].values.astype(bool)
    shift = np.where(late, LATE_RESTOCK_SHIFT, 0.0)
    take = np.minimum(shift, base[:, 1] + base[:, 2])           # cannot take more than refurbish+scrap hold
    frac_ref = np.divide(base[:, 1], base[:, 1] + base[:, 2], out=np.full(len(base), 0.5),
                         where=(base[:, 1] + base[:, 2]) > 0)
    p = base.copy()
    p[:, 0] += take
    p[:, 1] -= take * frac_ref
    p[:, 2] -= take * (1 - frac_ref)
    return p / p.sum(axis=1, keepdims=True)


def sample_dispositions(df: pd.DataFrame, seed: int) -> np.ndarray:
    """Simulated outcome for each ACTUAL return candidate: one draw from disposition_probs."""
    rng = np.random.default_rng(seed)
    p = disposition_probs(df)
    u = rng.random(len(df))[:, None]
    return (u > np.cumsum(p, axis=1)).sum(axis=1).clip(0, len(DISPOSITIONS) - 1)
