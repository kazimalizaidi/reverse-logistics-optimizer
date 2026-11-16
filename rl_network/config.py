"""
Central configuration. Every assumption lives here so each output number can be traced back
to one editable value.  Money is Brazilian reais (R$), distances km, weights kg, CO2 kg CO2e.

Legend used throughout the project:
  REAL       taken from the Olist data (orders, prices, weights, locations, delivery times)
  CALIBRATED fitted to Olist data at run time (freight rate, transit speed, handling days)
  PROXY      a stand-in for something Olist does not record (return = 1-2 star review)
  ASSUMED    invented for this project (disposition outcomes, facilities, costs, capacities)
"""
from dataclasses import dataclass, field
import numpy as np

SEED = 42

DISPOSITIONS = ["restock", "refurbish", "scrap"]
D = len(DISPOSITIONS)

# --------------------------------------------------------------------------- PROXY label
RETURN_SCORE_MAX = 2          # a delivered order with a review score <= 2 is a "return candidate"

# --------------------------------------------------------------------------- hubs (REAL cities)
# name, lat, lon   (collection hubs = regional demand centres)
HUBS = [
    ("São Paulo",      -23.5505, -46.6333),
    ("Rio de Janeiro", -22.9068, -43.1729),
    ("Belo Horizonte", -19.9167, -43.9345),
    ("Curitiba",       -25.4284, -49.2733),
    ("Porto Alegre",   -30.0346, -51.2177),
    ("Brasília",       -15.7939, -47.8828),
    ("Salvador",       -12.9777, -38.5016),
    ("Recife",          -8.0476, -34.8770),
    ("Belém",           -1.4558, -48.4902),
]

# Each customer state feeds its nearest regional hub (REAL states, regional grouping chosen here).
STATE_TO_HUB = {
    "SP": "São Paulo",
    "RJ": "Rio de Janeiro", "ES": "Rio de Janeiro",
    "MG": "Belo Horizonte",
    "PR": "Curitiba", "SC": "Curitiba",
    "RS": "Porto Alegre",
    "DF": "Brasília", "GO": "Brasília", "MT": "Brasília", "MS": "Brasília", "TO": "Brasília",
    "BA": "Salvador", "SE": "Salvador",
    "PE": "Recife", "PB": "Recife", "RN": "Recife", "AL": "Recife", "CE": "Recife",
    "PI": "Recife", "MA": "Recife",
    "PA": "Belém", "AM": "Belém", "AC": "Belém", "RO": "Belém", "RR": "Belém", "AP": "Belém",
}

# --------------------------------------------------------------------------- ASSUMED dispositions
# Olist records no disposition outcomes. These tables are ASSUMPTIONS, not findings:
# the share of returned parcels that end as [restock, refurbish, scrap], by product group.
CATEGORY_GROUP_KEYWORDS = {          # first match wins; keywords are matched in the English category name
    "electronics":   ["computers", "electronics", "telephony", "audio", "tablets", "pc_gamer",
                      "consoles", "signaling", "cine_photo", "dvds_blu"],
    "appliances":    ["appliances", "air_conditioning", "kitchen_dining"],
    "fashion":       ["fash", "luggage", "watches", "bags"],
    "beauty_health": ["health_beauty", "perfumery", "diapers", "hygiene"],
    "toys_baby":     ["toys", "baby"],
    "media":         ["books", "cds_dvds", "music", "stationery", "art", "cool_stuff"],
    "tools_auto":    ["auto", "tools", "industry", "agro", "construction"],
    "home":          ["furniture", "bed_bath", "housewares", "home_", "garden", "decor", "christmas",
                      "party", "pet_shop", "sports", "flowers", "la_cuisine", "market"],
}
DISPOSITION_MIX = {                  # [restock, refurbish, scrap]  (each row sums to 1)
    "electronics":   [0.35, 0.45, 0.20],
    "appliances":    [0.30, 0.45, 0.25],
    "fashion":       [0.65, 0.10, 0.25],
    "beauty_health": [0.35, 0.05, 0.60],
    "toys_baby":     [0.60, 0.15, 0.25],
    "media":         [0.75, 0.05, 0.20],
    "tools_auto":    [0.50, 0.35, 0.15],
    "home":          [0.45, 0.30, 0.25],
    "other":         [0.55, 0.20, 0.25],
}
# ASSUMED: when the original delivery was late, a return is more often about the delay than the
# item, so this share of the refurbish+scrap mass moves to restock.
LATE_RESTOCK_SHIFT = 0.15

# --------------------------------------------------------------------------- ASSUMED facilities
# Hypothetical processing sites in real cities. Capacities are expressed as a SHARE of the
# REFERENCE weekly return volume (the median weekly volume in the training period), so they are
# fixed physical capacities: a busier planning week genuinely stresses the network.
#   cap_share[d]  weekly capacity for disposition d (0 = cannot process d)
#   cap_total     overall weekly throughput cap
#   fixed_cost    weekly cost of keeping the site open (R$)
#   proc_cost[d]  variable processing cost per unit (R$)
#   yield_[d]     share of units reaching the intended outcome (1 - yield = scrap rate;
#                 failed units earn only salvage value)
FACILITIES = [
    dict(name="São Paulo Hub", city="São Paulo",
         cap_share=dict(restock=0.40, refurbish=0.18, scrap=0.12), cap_total=0.55,
         fixed_cost=2800, proc_cost=dict(restock=4.0, refurbish=36.0, scrap=6.0),
         yield_=dict(restock=0.98, refurbish=0.82, scrap=1.0)),
    dict(name="Belo Horizonte Refurb Center", city="Belo Horizonte",
         cap_share=dict(restock=0.20, refurbish=0.14, scrap=0.0), cap_total=0.30,
         fixed_cost=2100, proc_cost=dict(restock=3.5, refurbish=30.0, scrap=0.0),
         yield_=dict(restock=0.98, refurbish=0.88, scrap=1.0)),
    dict(name="Curitiba Restock DC", city="Curitiba",
         cap_share=dict(restock=0.22, refurbish=0.0, scrap=0.0), cap_total=0.22,
         fixed_cost=1400, proc_cost=dict(restock=3.0, refurbish=0.0, scrap=0.0),
         yield_=dict(restock=0.99, refurbish=1.0, scrap=1.0)),
    dict(name="Recife Recycling", city="Recife",
         cap_share=dict(restock=0.0, refurbish=0.06, scrap=0.14), cap_total=0.20,
         fixed_cost=1600, proc_cost=dict(restock=0.0, refurbish=27.0, scrap=4.0),
         yield_=dict(restock=1.0, refurbish=0.78, scrap=1.0)),
    dict(name="Brasília Multi-Site", city="Brasília",
         cap_share=dict(restock=0.10, refurbish=0.08, scrap=0.08), cap_total=0.22,
         fixed_cost=1200, proc_cost=dict(restock=4.0, refurbish=26.0, scrap=5.0),
         yield_=dict(restock=0.97, refurbish=0.80, scrap=1.0)),
]


@dataclass
class Econ:
    """Economic / environmental parameters shared by optimizer and evaluator.
    the shipping and transit parameters are overwritten at run time by values CALIBRATED
    on Olist (see calibration.py); the defaults below are fallbacks."""
    # shipping cost per parcel = base + per_km * km + per_kg_km * kg * km        (all CALIBRATED on Olist
    # freight; the defaults are placeholders used by unit tests, the pipeline overwrites them)
    ship_base_cost: float = 0.0              # R$ per parcel
    ship_cost_per_km: float = 0.0            # R$ per km
    ship_cost_per_kg_km: float = 0.0030      # R$ per kg per km
    co2_per_kg_km: float = 0.00012           # kg CO2e per kg per km   (road freight, literature value)
    proc_co2: dict = field(default_factory=lambda: dict(restock=0.15, refurbish=1.4, scrap=0.7))
    carbon_price: float = 0.10               # R$ per kg CO2e (internal shadow price, ASSUMED)
    sla_days: dict = field(default_factory=lambda: dict(restock=8.0, refurbish=14.0, scrap=21.0))
    late_penalty: float = 1.50               # R$ per unit per day beyond SLA (ASSUMED)
    recovery_rate: dict = field(default_factory=lambda: dict(restock=0.92, refurbish=0.55, scrap=0.05))
    salvage_rate: float = 0.05               # value share for units failing processing
    landfill_fee: float = 8.0                # R$ per unit sent to disposal
    landfill_co2: float = 2.5                # kg CO2e per unit sent to disposal
    min_util: float = 0.10                   # open site must use >= this share of cap_total
    cap_buffer: float = 0.05                 # plan to use at most (1 - buffer) of capacity
    road_factor: float = 1.3                 # straight-line -> road distance
    # transit days of a leg = base + coef * sqrt(km)                            (CALIBRATED on Olist
    # carrier-to-customer times; defaults are placeholders, the pipeline overwrites them)
    transit_base_days: float = 0.5
    transit_sqrt_coef: float = 0.045
    min_local_km: float = 15.0

    def sla_vec(self):
        return np.array([self.sla_days[d] for d in DISPOSITIONS])

    def rate_vec(self):
        return np.array([self.recovery_rate[d] for d in DISPOSITIONS])

    def proc_co2_vec(self):
        return np.array([self.proc_co2[d] for d in DISPOSITIONS])
