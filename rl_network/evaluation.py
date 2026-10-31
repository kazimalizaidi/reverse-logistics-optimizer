"""
Evaluate a routing plan against the REALIZED week.

A plan is built from an *estimated* demand (predicted, prior-based, ...). Reality then
delivers different volumes. We replay reality through the plan's routing SHARES:

    share[h,f,d] = planned_flow[h,f,d] / planned_volume[h,d]
    realized_flow[h,f,d] = share[h,f,d] * actual_volume[h,d]

If a facility is then loaded beyond its capacity, the overflow is the facility's BACKLOG;
for costing we conservatively assume backlogged units are diverted to disposal this week.
Closed facilities receive nothing (the plan's decision stands).
"""
import numpy as np
import pandas as pd

from .config import DISPOSITIONS, Econ, D
from .demand import Demand
from .network import Network
from .optimizer import Plan, arc_components, landfill_unit_cost


def realize(plan: Plan, plan_dem: Demand, real_dem: Demand, net: Network):
    """Returns (realized_flows H x F x D, realized_landfill H x D, backlog F, raw_load F)."""
    share = np.divide(plan.flows, plan_dem.vol[:, None, :],
                      out=np.zeros_like(plan.flows), where=plan_dem.vol[:, None, :] > 1e-12)
    flows = share * real_dem.vol[:, None, :]
    raw_load = flows.sum(axis=(0, 2))

    load_fd = flows.sum(axis=0)                                    # F x D
    cap_d = net.cap_d
    s_fd = np.where(load_fd > cap_d, np.divide(cap_d, load_fd, out=np.zeros_like(load_fd),
                                               where=load_fd > 0), 1.0)
    flows = flows * s_fd[None, :, :]
    load_f = flows.sum(axis=(0, 2))
    s_f = np.where(load_f > net.cap_total, net.cap_total / np.maximum(load_f, 1e-12), 1.0)
    flows = flows * s_f[None, :, None]
    final_load = flows.sum(axis=(0, 2))
    backlog = raw_load - final_load
    landfill = np.maximum(real_dem.vol - flows.sum(axis=1), 0.0)
    landfill[landfill < 1e-3] = 0.0          # drop solver round-off dust (< 0.001 unit)
    return flows, landfill, backlog, raw_load


def evaluate(plan: Plan, plan_dem: Demand, real_dem: Demand, net: Network, econ: Econ):
    flows, landfill, backlog, raw_load = realize(plan, plan_dem, real_dem, net)
    ac = arc_components(net, real_dem, econ)     # realized price / weight
    units = float(real_dem.vol.sum())

    shipping = float((ac["ship"] * flows).sum())
    processing = float((ac["proc"] * flows).sum())
    late_pen = float((ac["late"] * flows).sum())
    carbon_cost_arcs = float((ac["carbon"] * flows).sum())
    fixed = float((net.fixed_cost * plan.open_).sum())
    landfill_units = float(landfill.sum())
    landfill_fee = econ.landfill_fee * landfill_units
    landfill_carbon = econ.carbon_price * econ.landfill_co2 * landfill_units
    recovered = float((ac["value"] * flows).sum())
    original_value = float((real_dem.vol * real_dem.avg_price).sum())
    co2_kg = float((ac["co2"] * flows).sum()) + econ.landfill_co2 * landfill_units
    carbon_cost = carbon_cost_arcs + landfill_carbon
    net_cost = shipping + processing + late_pen + fixed + landfill_fee + carbon_cost - recovered
    breach_units = float((flows * (ac["late_days"] > 0)).sum())
    late_unit_days = float((flows * ac["late_days"]).sum())
    processed = float(flows.sum())

    summary = dict(
        scenario=plan.name,
        facilities_open=int(plan.open_.sum()),
        units_returned=units,
        units_processed=processed,
        units_to_disposal=landfill_units,
        landfill_diversion_rate=1 - landfill_units / units,
        backlog_units=float(backlog.sum()),
        sla_breach_units=breach_units,
        sla_breach_rate=breach_units / max(processed, 1e-12),
        avg_transit_days=float((flows * net.transit_days[:, :, None]).sum() / max(processed, 1e-12)),
        shipping_cost=shipping, processing_cost=processing, sla_penalty=late_pen,
        fixed_cost=fixed, disposal_fee=landfill_fee, carbon_cost=carbon_cost,
        recovered_value=recovered, original_value=original_value,
        value_recovery_rate=recovered / original_value,
        net_cost=net_cost,                       # negative = net profit (value recovered > costs)
        net_contribution=-net_cost,              # positive = net profit
        cost_to_serve=net_cost + recovered,     # all costs, before crediting recovered value
        net_cost_per_unit=net_cost / units,
        operating_cost_ex_carbon=net_cost - carbon_cost,
        co2_kg=co2_kg,
        co2_kg_per_unit=co2_kg / units,
        planned_objective=plan.objective,
        solver_status=plan.status,
    )

    fac_rows = []
    for f, fname in enumerate(net.facilities):
        ff = flows[:, f, :]
        fac_rows.append(dict(
            scenario=plan.name, facility=fname, open=int(plan.open_[f]),
            units_assigned=float(raw_load[f]), units_processed=float(ff.sum()),
            capacity=float(net.cap_total[f]),
            utilization=float(ff.sum() / net.cap_total[f]),
            backlog_units=float(backlog[f]),
            backlog_days=float(backlog[f] / (net.cap_total[f] / 7)),   # days of full-rate work
            sla_breach_units=float((ff * (ac["late_days"][:, f, :] > 0)).sum()),
            recovered_value=float((ac["value"][:, f, :] * ff).sum()),
            variable_cost=float(((ac["ship"] + ac["proc"] + ac["late"] + ac["carbon"])[:, f, :] * ff).sum()),
            fixed_cost=float(net.fixed_cost[f] * plan.open_[f]),
            **{f"units_{d}": float(ff[:, j].sum()) for j, d in enumerate(DISPOSITIONS)},
        ))
    fac = pd.DataFrame(fac_rows)
    fac["cost_to_serve"] = fac["variable_cost"] + fac["fixed_cost"]
    fac["recapture_margin"] = np.where(fac["recovered_value"] > 0,
                                       (fac["recovered_value"] - fac["cost_to_serve"]) / fac["recovered_value"].where(fac["recovered_value"] > 0, 1),
                                       np.nan)
    # disposal costs are not attributable to a facility; exported so totals reconcile
    summary["disposal_cost"] = landfill_fee + landfill_carbon
    return summary, fac, flows, landfill


def routing_table(plan: Plan, net: Network, econ: Econ) -> pd.DataFrame:
    rows = []
    for h in range(net.H):
        for f in range(net.F):
            for d in range(D):
                q = plan.flows[h, f, d]
                if q > 1e-6:
                    rows.append(dict(
                        scenario=plan.name, hub=net.hubs[h], facility=net.facilities[f],
                        disposition=DISPOSITIONS[d], planned_units=round(float(q), 2),
                        distance_km=round(float(net.dist_km[h, f]), 1),
                        transit_days=round(float(net.transit_days[h, f]), 2),
                        sla_days=econ.sla_days[DISPOSITIONS[d]],
                        late=bool(net.transit_days[h, f] > econ.sla_days[DISPOSITIONS[d]])))
    for h in range(net.H):
        for d in range(D):
            if plan.landfill[h, d] > 1e-6:
                rows.append(dict(scenario=plan.name, hub=net.hubs[h], facility="DISPOSAL",
                                 disposition=DISPOSITIONS[d],
                                 planned_units=round(float(plan.landfill[h, d]), 2),
                                 distance_km=np.nan, transit_days=np.nan,
                                 sla_days=econ.sla_days[DISPOSITIONS[d]], late=False))
    return pd.DataFrame(rows)


def realized_lane_table(plan: Plan, flows: np.ndarray, real_dem: Demand, net: Network, econ: Econ) -> pd.DataFrame:
    """One row per realized (hub, facility, disposition) lane with units, lateness, value and cost.
    Feeds the SLA-breach and recapture-margin visuals; sums reconcile to scenario_summary."""
    ac = arc_components(net, real_dem, econ)
    rows = []
    for h in range(net.H):
        for f in range(net.F):
            for d in range(D):
                q = float(flows[h, f, d])
                if q <= 1e-6:
                    continue
                late_days = float(ac["late_days"][h, f, d])
                rows.append(dict(
                    scenario=plan.name, hub=net.hubs[h], facility=net.facilities[f],
                    disposition=DISPOSITIONS[d], lane=f"{net.hubs[h]} \u2192 {net.facilities[f].split()[0]}",
                    units=q, distance_km=float(net.dist_km[h, f]),
                    transit_days=float(net.transit_days[h, f]), sla_days=econ.sla_days[DISPOSITIONS[d]],
                    late_days=late_days, sla_breach_units=q if late_days > 0 else 0.0,
                    recovered_value=float(ac["value"][h, f, d] * q),
                    variable_cost=float((ac["ship"] + ac["proc"] + ac["late"] + ac["carbon"])[h, f, d] * q)))
    return pd.DataFrame(rows)
