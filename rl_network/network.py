"""Network data structure + builder from config."""
from dataclasses import dataclass
import numpy as np
from .config import HUBS, FACILITIES, DISPOSITIONS, Econ


@dataclass
class Network:
    hubs: list
    facilities: list
    dist_km: np.ndarray        # H x F
    transit_days: np.ndarray   # H x F
    cap_d: np.ndarray          # F x D  weekly capacity per disposition
    cap_total: np.ndarray      # F      weekly overall throughput cap
    fixed_cost: np.ndarray     # F      weekly fixed cost if open
    proc_cost: np.ndarray      # F x D
    yield_: np.ndarray         # F x D

    @property
    def H(self):
        return len(self.hubs)

    @property
    def F(self):
        return len(self.facilities)

    @property
    def supports(self):
        return self.cap_d > 0   # F x D boolean


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def build_network(total_weekly_volume: float, econ: Econ) -> Network:
    city_xy = {h[0]: (h[1], h[2]) for h in HUBS}
    hubs = [h[0] for h in HUBS]
    facs = [f["name"] for f in FACILITIES]
    H, F = len(hubs), len(facs)

    dist = np.zeros((H, F))
    for i, h in enumerate(hubs):
        for j, f in enumerate(FACILITIES):
            lat1, lon1 = city_xy[h]
            lat2, lon2 = city_xy[f["city"]]
            km = haversine_km(lat1, lon1, lat2, lon2) * econ.road_factor
            dist[i, j] = max(km, econ.min_local_km)
    transit = econ.handling_days + dist / econ.speed_km_day

    cap_d = np.array([[f["cap_share"][d] for d in DISPOSITIONS] for f in FACILITIES]) * total_weekly_volume
    cap_total = np.array([f["cap_total"] for f in FACILITIES]) * total_weekly_volume
    fixed = np.array([f["fixed_cost"] for f in FACILITIES], dtype=float)
    proc = np.array([[f["proc_cost"][d] for d in DISPOSITIONS] for f in FACILITIES])
    yld = np.array([[f["yield_"][d] for d in DISPOSITIONS] for f in FACILITIES])
    return Network(hubs, facs, dist, transit, cap_d, cap_total, fixed, proc, yld)
