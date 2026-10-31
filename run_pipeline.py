"""
End-to-end pipeline:  data -> classifier -> demand estimates -> MILP -> evaluation -> CSV/JSON exports

Usage:  python run_pipeline.py [--n 30000] [--seed 42] [--out outputs]
All outputs (consumed later by the Returns Control Tower dashboard) land in --out.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from rl_network import classifier as clf
from rl_network import demand as dm
from rl_network.config import DISPOSITIONS, Econ, SEED, WEEK_CATEGORY_MIX
from rl_network.data_generation import generate_returns
from rl_network.evaluation import evaluate, realized_lane_table, routing_table
from rl_network.network import build_network
from rl_network.optimizer import greedy_nearest, solve_milp


def main(n: int, seed: int, out: Path, carbon_prices=(0.0, 0.10, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0)):
    out.mkdir(parents=True, exist_ok=True)
    econ = Econ()

    # 1. Data: train on history, plan on the held-out "incoming week"
    n_week = int(round(n * 0.30))
    train_df = generate_returns(n - n_week, seed)                                   # history
    week_df = generate_returns(n_week, seed + 1, category_mix=WEEK_CATEGORY_MIX)    # promo week
    print(f"history={len(train_df):,} returns | planning week={len(week_df):,} returns")

    # 2. Classifier
    raw, cal = clf.train(train_df, seed)
    y = week_df["disposition_idx"].values
    true_p = week_df[[f"_true_p_{d}" for d in DISPOSITIONS]].values
    prior = np.bincount(train_df["disposition_idx"], minlength=3) / len(train_df)
    p_raw = raw.predict_proba(week_df[clf.FEATURES])
    p_cal = cal.predict_proba(week_df[clf.FEATURES])
    metrics = {"ensemble_uncalibrated": clf.evaluate(p_raw, y, true_p, prior),
               "ensemble_calibrated": clf.evaluate(p_cal, y, true_p, prior),
               "training_class_prior": dict(zip(DISPOSITIONS, prior.round(4).tolist()))}
    json.dump(metrics, open(out / "classifier_metrics.json", "w"), indent=2)
    clf.feature_importance(raw, week_df, seed).to_csv(out / "feature_importance.csv", index=False)
    probs = p_cal   # the optimizer consumes calibrated probabilities

    scored = week_df.drop(columns=[c for c in week_df if c.startswith("_true_p_")]).copy()
    for j, d in enumerate(DISPOSITIONS):
        scored[f"p_{d}"] = probs[:, j]
    scored["predicted_disposition"] = [DISPOSITIONS[i] for i in probs.argmax(axis=1)]
    scored["prediction_confidence"] = probs.max(axis=1)
    scored.to_csv(out / "returns_scored.csv", index=False)

    # 3. Network sized to the planning week, and the four demand views
    net = build_network(len(week_df), econ)
    H = net.H
    real = dm.from_labels(week_df, H)
    est_pred = dm.from_probs(week_df, probs, H)
    est_prior = dm.from_prior(week_df, prior, H)

    vol_cmp = []
    for h in range(H):
        for j, d in enumerate(DISPOSITIONS):
            vol_cmp.append(dict(hub=net.hubs[h], disposition=d, actual=real.vol[h, j],
                                predicted_expected=est_pred.vol[h, j],
                                prior_based=est_prior.vol[h, j]))
    vol_cmp = pd.DataFrame(vol_cmp)
    wape = lambda col: float((vol_cmp[col] - vol_cmp["actual"]).abs().sum() / vol_cmp["actual"].sum())
    metrics["volume_forecast_wape_hub_x_disposition"] = {
        "classifier_expected": wape("predicted_expected"), "prior_based": wape("prior_based")}
    json.dump(metrics, open(out / "classifier_metrics.json", "w"), indent=2)
    vol_cmp.to_csv(out / "volume_forecast_vs_actual.csv", index=False)

    # 4. Plans: baseline, no-classifier MILP, classifier MILP, oracle MILP
    scenarios = [
        ("A_greedy_nearest",   greedy_nearest(net, est_pred, econ, "A_greedy_nearest"), est_pred),
        ("B_milp_prior_only",  solve_milp(net, est_prior, econ, "B_milp_prior_only"), est_prior),
        ("C_milp_classifier",  solve_milp(net, est_pred, econ, "C_milp_classifier"), est_pred),
        # perfect-information lower bound: true volumes AND no planning buffer, so it faces
        # exactly the constraints of the realized world
        ("D_milp_oracle",      solve_milp(net, real, replace(econ, cap_buffer=0.0), "D_milp_oracle"), real),
    ]
    summaries, facs, routes, lanes = [], [], [], []
    for name, plan, plan_dem in scenarios:
        s, f, flows_real, _ = evaluate(plan, plan_dem, real, net, econ)
        summaries.append(s); facs.append(f); routes.append(routing_table(plan, net, econ))
        lanes.append(realized_lane_table(plan, flows_real, real, net, econ))
        print(f"{name:20s} status={plan.status:10s} open={int(plan.open_.sum())} "
              f"net_cost=${s['net_cost']:,.0f} co2={s['co2_kg']:,.0f}kg "
              f"backlog={s['backlog_units']:,.0f} sla_breach={s['sla_breach_units']:,.0f}")
    pd.DataFrame(summaries).to_csv(out / "scenario_summary.csv", index=False)
    pd.concat(facs).to_csv(out / "facility_kpis.csv", index=False)
    pd.concat(routes).to_csv(out / "routing_plan.csv", index=False)
    pd.concat(lanes).to_csv(out / "realized_routing.csv", index=False)
    pd.DataFrame({"scenario": [x[0] for x in scenarios],
                  "scenario_label": ["A  Nearest-facility", "B  No classifier", "C  Classifier + MILP",
                                     "D  Perfect information"],
                  "sort_order": [1, 2, 3, 4]}).to_csv(out / "dim_scenario.csv", index=False)

    # 5. Carbon shadow-price sweep = marginal abatement cost curve for the classifier-driven MILP.
    #    Realistic carbon prices (~$0.1/kg) barely move routing because freight is a small share of
    #    total cost; the wide sweep shows what it WOULD cost to push emissions lower.
    sens = []
    for cp in carbon_prices:
        e = Econ(carbon_price=cp)
        plan = solve_milp(net, est_pred, e, f"C_milp_classifier@{cp}")
        s, _, _, _ = evaluate(plan, est_pred, real, net, e)
        sens.append(dict(carbon_price_usd_per_kg=cp, facilities_open=s["facilities_open"],
                         operating_cost_ex_carbon=s["operating_cost_ex_carbon"],
                         co2_kg=s["co2_kg"], co2_kg_per_unit=s["co2_kg_per_unit"],
                         landfill_diversion_rate=s["landfill_diversion_rate"],
                         value_recovery_rate=s["value_recovery_rate"]))
    pd.DataFrame(sens).to_csv(out / "carbon_sensitivity.csv", index=False)

    # 6. Static network description for the dashboard map
    pd.DataFrame({"facility": net.facilities, "weekly_capacity": net.cap_total,
                  "fixed_cost_weekly": net.fixed_cost,
                  **{f"cap_{d}": net.cap_d[:, j] for j, d in enumerate(DISPOSITIONS)}}
                 ).to_csv(out / "network_facilities.csv", index=False)
    pd.DataFrame(net.dist_km, index=net.hubs, columns=net.facilities).round(1).to_csv(
        out / "network_distance_km.csv")
    json.dump({"econ": econ.__dict__, "seed": seed, "n_returns": n}, open(out / "run_config.json", "w"),
              indent=2, default=str)
    print(f"\nWrote outputs to {out.resolve()}")
    return pd.DataFrame(summaries), metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    a = ap.parse_args()
    main(a.n, a.seed, a.out)
