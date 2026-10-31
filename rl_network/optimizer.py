"""
Routing optimizer (MILP) and nearest-facility greedy baseline.

Sets
  h in Hubs, f in Facilities, d in Dispositions {restock, refurbish, scrap}

Decision variables
  x[h,f,d] >= 0   units of disposition d moved from hub h to facility f
                  (only defined where facility f can process d)
  u[h,d]   >= 0   units sent to landfill / third-party disposal (always feasible)
  y[f] in {0,1}   1 if facility f is open this week

Per-unit arc cost  c[h,f,d] =  shipping + processing + SLA lateness penalty
                              + carbon_price * CO2  -  recovered value
  shipping  = ship_rate * avg_weight[h,d] * dist[h,f]
  lateness  = late_penalty * max(0, transit[h,f] - SLA[d])
  CO2       = co2_per_kg_km * avg_weight[h,d] * dist[h,f] + proc_co2[d]
  value     = avg_price[h,d] * ( yield[f,d]*rate[d] + (1-yield[f,d])*salvage )

Objective   min  sum c[h,f,d] x[h,f,d] + sum L[h,d] u[h,d] + sum fixed[f] y[f]
  with L[h,d] = landfill_fee + carbon_price*landfill_co2   (no value recovered)

Constraints
  (1) flow conservation   sum_f x[h,f,d] + u[h,d] = vol[h,d]          for all h,d
  (2) per-disposition cap sum_h x[h,f,d] <= (1-b)*cap_d[f,d] * y[f]     for all f,d
  (3) total throughput    sum_{h,d} x[h,f,d] <= (1-b)*cap_total[f] * y[f] for all f
  (4) minimum utilisation sum_{h,d} x[h,f,d] >= min_util*cap_total[f]*y[f]
  where b = econ.cap_buffer is a planning safety margin (headroom for forecast error).
"""
from dataclasses import dataclass, field
import numpy as np
import pulp

from .config import Econ, D, DISPOSITIONS
from .demand import Demand
from .network import Network


@dataclass
class Plan:
    name: str
    flows: np.ndarray        # H x F x D planned units
    landfill: np.ndarray     # H x D planned units
    open_: np.ndarray        # F  0/1
    status: str = "n/a"
    objective: float = float("nan")   # planned (expected-volume) objective, USD
    solve_seconds: float = float("nan")


def arc_components(net: Network, dem: Demand, econ: Econ) -> dict:
    """Per-unit economics of every (h,f,d) arc, as separate H x F x D arrays so the
    optimizer and the evaluator use the identical formulas."""
    dist = net.dist_km[:, :, None]
    w = dem.avg_weight[:, None, :]
    p = dem.avg_price[:, None, :]
    ship = econ.ship_cost_per_kg_km * w * dist
    proc = np.broadcast_to(net.proc_cost[None, :, :], ship.shape).copy()
    late_days = np.maximum(0.0, net.transit_days[:, :, None] - econ.sla_vec()[None, None, :])
    late = econ.late_penalty * late_days * np.ones_like(ship)
    co2 = econ.co2_per_kg_km * w * dist + econ.proc_co2_vec()[None, None, :]
    carbon = econ.carbon_price * co2
    yld = net.yield_[None, :, :]
    value = p * (yld * econ.rate_vec()[None, None, :] + (1 - yld) * econ.salvage_rate)
    cost = ship + proc + late + carbon - value
    return dict(ship=ship, proc=proc, late=late, late_days=late_days * np.ones_like(ship),
                co2=co2 * np.ones_like(ship), carbon=carbon, value=value * np.ones_like(ship),
                cost=cost)


def landfill_unit_cost(econ: Econ) -> float:
    return econ.landfill_fee + econ.carbon_price * econ.landfill_co2


def _get_solver(time_limit: int):
    avail = pulp.listSolvers(onlyAvailable=True)
    for name in ("PULP_CBC_CMD", "HiGHS", "COIN_CMD"):
        if name in avail:
            return pulp.getSolver(name, msg=False, timeLimit=time_limit)
    raise RuntimeError(f"No MILP solver available to PuLP (found {avail}). "
                       "Run: pip install highspy")


def solve_milp(net: Network, dem: Demand, econ: Econ, name: str = "milp",
               time_limit: int = 120) -> Plan:
    import time
    H, F = net.H, net.F
    ac = arc_components(net, dem, econ)
    cost = ac["cost"]
    sup = net.supports

    prob = pulp.LpProblem(name, pulp.LpMinimize)
    x = {(h, f, d): pulp.LpVariable(f"x_{h}_{f}_{d}", lowBound=0)
         for h in range(H) for f in range(F) for d in range(D) if sup[f, d]}
    u = {(h, d): pulp.LpVariable(f"u_{h}_{d}", lowBound=0) for h in range(H) for d in range(D)}
    y = {f: pulp.LpVariable(f"y_{f}", cat=pulp.LpBinary) for f in range(F)}

    L = landfill_unit_cost(econ)
    prob += (pulp.lpSum(cost[h, f, d] * v for (h, f, d), v in x.items())
             + pulp.lpSum(L * v for v in u.values())
             + pulp.lpSum(net.fixed_cost[f] * y[f] for f in range(F)))

    for h in range(H):
        for d in range(D):
            prob += (pulp.lpSum(x[h, f, d] for f in range(F) if sup[f, d]) + u[h, d]
                     == float(dem.vol[h, d])), f"flow_{h}_{d}"
    for f in range(F):
        for d in range(D):
            if sup[f, d]:
                prob += (pulp.lpSum(x[h, f, d] for h in range(H))
                         <= float(net.cap_d[f, d]) * (1 - econ.cap_buffer) * y[f]), f"capd_{f}_{d}"
        tot = pulp.lpSum(x[h, f, d] for h in range(H) for d in range(D) if sup[f, d])
        prob += tot <= float(net.cap_total[f]) * (1 - econ.cap_buffer) * y[f], f"captot_{f}"
        prob += tot >= econ.min_util * float(net.cap_total[f]) * y[f], f"minutil_{f}"

    t0 = time.time()
    prob.solve(_get_solver(time_limit))
    secs = time.time() - t0

    flows = np.zeros((H, F, D))
    for (h, f, d), v in x.items():
        flows[h, f, d] = max(0.0, v.value() or 0.0)
    landfill = np.array([[max(0.0, u[h, d].value() or 0.0) for d in range(D)] for h in range(H)])
    open_ = np.array([round(y[f].value() or 0.0) for f in range(F)])
    return Plan(name, flows, landfill, open_, pulp.LpStatus[prob.status],
                float(pulp.value(prob.objective)), secs)


def greedy_nearest(net: Network, dem: Demand, econ: Econ, name: str = "greedy_nearest") -> Plan:
    """Baseline: every facility open; each (hub, disposition) stream goes to the NEAREST
    facility that can process it until that facility's capacity is used, then to the
    next nearest. Streams are served largest-first. Leftover -> disposal."""
    H, F = net.H, net.F
    sup = net.supports
    rem_d = net.cap_d * (1 - econ.cap_buffer)
    rem_t = net.cap_total * (1 - econ.cap_buffer)
    flows = np.zeros((H, F, D))
    landfill = np.zeros((H, D))
    streams = sorted(((h, d) for h in range(H) for d in range(D)),
                     key=lambda hd: -dem.vol[hd])
    for h, d in streams:
        need = dem.vol[h, d]
        for f in sorted((f for f in range(F) if sup[f, d]), key=lambda f: net.dist_km[h, f]):
            take = min(need, rem_d[f, d], rem_t[f])
            if take > 0:
                flows[h, f, d] += take
                rem_d[f, d] -= take
                rem_t[f] -= take
                need -= take
            if need <= 1e-9:
                break
        landfill[h, d] = max(0.0, need)

    ac = arc_components(net, dem, econ)
    obj = (float((ac["cost"] * flows).sum()) + landfill_unit_cost(econ) * float(landfill.sum())
           + float(net.fixed_cost.sum()))
    return Plan(name, flows, landfill, np.ones(F, dtype=int), "Heuristic", obj, 0.0)
