"""
Central configuration: every economic / network assumption lives here so that
each number in the outputs can be traced back to a single editable value.

All monetary values are USD, distances km, weights kg, emissions kg CO2e.
The Pakistani city network is ILLUSTRATIVE (real coordinates, invented capacities
and costs). Replace with your own network data to apply this to a real case.
"""
from dataclasses import dataclass, field
import numpy as np

SEED = 42

DISPOSITIONS = ["restock", "refurbish", "scrap"]
D = len(DISPOSITIONS)

CATEGORIES = ["electronics", "appliances", "home", "apparel", "beauty", "toys"]
REASONS = ["changed_mind", "size_fit", "wrong_item", "not_as_described",
           "defective", "damaged_in_transit"]
CONDITIONS = ["intact", "opened", "damaged"]

# Planning-week category mix (order = CATEGORIES). A promo-driven electronics/appliance
# peak, so a forecast built only from historical class mix is biased.
WEEK_CATEGORY_MIX = [0.34, 0.10, 0.12, 0.20, 0.10, 0.14]

# name, lat, lon, share of returns, remoteness (extra transit days customer -> hub)
HUBS = [
    ("Karachi",    24.8607, 67.0011, 0.26, 0.4),
    ("Lahore",     31.5204, 74.3587, 0.22, 0.4),
    ("Islamabad",  33.6844, 73.0479, 0.12, 0.5),
    ("Faisalabad", 31.4504, 73.1350, 0.09, 0.6),
    ("Multan",     30.1575, 71.4724, 0.07, 0.8),
    ("Peshawar",   34.0151, 71.5249, 0.09, 0.9),
    ("Hyderabad",  25.3960, 68.3578, 0.06, 0.7),
    ("Quetta",     30.1798, 66.9750, 0.04, 1.8),
    ("Sialkot",    32.4945, 74.5229, 0.05, 0.8),
]

# Facilities. Capacities are SHARES of the planning-week return volume so the
# network scales with whatever batch is simulated.
#   cap_share[d] weekly capacity for disposition d (0 = cannot process d)
#   cap_total    overall weekly throughput cap
#   fixed_cost   weekly cost of keeping the site open (USD)
#   proc_cost[d] variable processing cost per unit (USD)
#   yield_[d]    share of units reaching the intended outcome (1 - yield = scrap
#                rate; failed units earn only salvage value)
FACILITIES = [
    dict(name="Karachi Hub", city="Karachi",
         cap_share=dict(restock=0.22, refurbish=0.10, scrap=0.08), cap_total=0.34,
         fixed_cost=14000, proc_cost=dict(restock=1.8, refurbish=13.5, scrap=2.4),
         yield_=dict(restock=0.98, refurbish=0.82, scrap=1.0)),
    dict(name="Lahore Refurb Center", city="Lahore",
         cap_share=dict(restock=0.24, refurbish=0.14, scrap=0.0), cap_total=0.34,
         fixed_cost=12000, proc_cost=dict(restock=1.5, refurbish=11.0, scrap=0.0),
         yield_=dict(restock=0.98, refurbish=0.88, scrap=1.0)),
    dict(name="Islamabad Restock DC", city="Islamabad",
         cap_share=dict(restock=0.18, refurbish=0.0, scrap=0.0), cap_total=0.18,
         fixed_cost=7000, proc_cost=dict(restock=1.2, refurbish=0.0, scrap=0.0),
         yield_=dict(restock=0.99, refurbish=1.0, scrap=1.0)),
    dict(name="Faisalabad Recycling", city="Faisalabad",
         cap_share=dict(restock=0.0, refurbish=0.06, scrap=0.12), cap_total=0.18,
         fixed_cost=9000, proc_cost=dict(restock=0.0, refurbish=10.0, scrap=1.6),
         yield_=dict(restock=1.0, refurbish=0.78, scrap=1.0)),
    dict(name="Multan Multi-Site", city="Multan",
         cap_share=dict(restock=0.08, refurbish=0.10, scrap=0.07), cap_total=0.22,
         fixed_cost=6500, proc_cost=dict(restock=1.6, refurbish=9.0, scrap=2.0),
         yield_=dict(restock=0.97, refurbish=0.80, scrap=1.0)),
]


@dataclass
class Econ:
    """Economic / environmental parameters shared by optimizer and evaluator."""
    ship_cost_per_kg_km: float = 0.0030
    co2_per_kg_km: float = 0.00012
    proc_co2: dict = field(default_factory=lambda: dict(restock=0.15, refurbish=1.4, scrap=0.7))
    carbon_price: float = 0.10               # USD per kg CO2e (internal shadow price)
    sla_days: dict = field(default_factory=lambda: dict(restock=2.0, refurbish=4.0, scrap=7.0))
    late_penalty: float = 0.75               # USD per unit per day beyond SLA
    recovery_rate: dict = field(default_factory=lambda: dict(restock=0.92, refurbish=0.55, scrap=0.05))
    salvage_rate: float = 0.05               # value share for units failing processing
    landfill_fee: float = 6.0                # USD per unit sent to disposal
    landfill_co2: float = 2.5                # kg CO2e per unit sent to disposal
    min_util: float = 0.10                   # open site must use >= this share of cap_total
    road_factor: float = 1.3                 # haversine -> road distance
    speed_km_day: float = 550.0
    handling_days: float = 0.5
    min_local_km: float = 15.0
    cap_buffer: float = 0.05                 # plan to use at most (1-buffer) of capacity

    def sla_vec(self):
        return np.array([self.sla_days[d] for d in DISPOSITIONS])

    def rate_vec(self):
        return np.array([self.recovery_rate[d] for d in DISPOSITIONS])

    def proc_co2_vec(self):
        return np.array([self.proc_co2[d] for d in DISPOSITIONS])
