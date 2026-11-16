"""
End-to-end pipeline on REAL Olist data:
  load -> order table -> [calibrate freight/transit] -> return classifier -> demand estimates
       -> MILP + baselines -> realized-week evaluation -> exports  (+ rolling backtest)

Usage:
  python run_pipeline.py                         # featured week = the peak-return week
  python run_pipeline.py --week 2018-04-02       # any Monday present in the data
  python run_pipeline.py --no-backtest           # skip the rolling backtest (faster)

Place the Olist CSV files in data/olist/ first (see README). Outputs land in --out and are
the only files the Streamlit dashboard reads.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from rl_network import classifier as clf
from rl_network import demand as dm
from rl_network.config import DISPOSITIONS, HUBS, RETURN_SCORE_MAX, SEED, Econ
from rl_network.olist_data import build_order_table, disposition_probs, load_raw
from rl_network.optimizer import solve_milp
from rl_network.evaluation import evaluate
from rl_network.planning import eligible_weeks, peak_week, plan_week, prepare

CARBON_PRICES = (0.0, 0.10, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0)
SCEN_LABELS = {"A_greedy_nearest": "A  Nearest-facility", "B_milp_prior_only": "B  No classifier",
               "C_milp_classifier": "C  Classifier + MILP", "D_milp_oracle": "D  Perfect information"}


def wape(pred, actual):
    return float(np.abs(pred - actual).sum() / max(actual.sum(), 1e-9))


def run_featured(df, week, out: Path, seed: int):
    econ0 = Econ()
    prep = prepare(df, week, econ0, seed, fast=False)
    res = plan_week(df, week, prep, seed)
    week_df, train_df, real, net, econ = res["week_df"], res["train_df"], res["real"], res["net"], res["econ"]
    p_cal = res["p"]
    p_raw = clf.predict(prep.raw_model, week_df)
    y = week_df[clf.TARGET].values

    # ---- classifier quality on the held-out week (real labels, strictly out-of-time) ----
    metrics = {"planning_week": str(week.date()), "label": f"return candidate = review score <= {RETURN_SCORE_MAX}",
               "n_train_orders": int(len(train_df)), "n_week_orders": int(len(week_df)),
               "train_return_rate": float(train_df[clf.TARGET].mean()),
               "week_return_rate": float(y.mean()),
               "calibrated": clf.evaluate(p_cal, week_df, train_df),
               "uncalibrated": clf.evaluate(p_raw, week_df, train_df)}
    bl = clf.baselines(train_df, week_df)
    metrics["expected_returns"] = {"actual": int(y.sum()), "classifier": float(p_cal.sum()),
                                   **{k: float(v.sum()) for k, v in bl.items()}}
    metrics["returns_forecast_error_pct"] = {k: float((v - y.sum()) / y.sum() * 100)
                                             for k, v in {"classifier": p_cal.sum(), **{k: v.sum() for k, v in bl.items()}}.items()}
    clf.feature_importance(prep.raw_model, week_df, seed).to_csv(out / "feature_importance.csv", index=False)
    clf.calibration_table(p_cal, y).to_csv(out / "calibration_table.csv", index=False)

    # ---- demand forecast vs actual, hub x disposition ----
    H = len(HUBS)
    rows = []
    for h in range(H):
        for j, d in enumerate(DISPOSITIONS):
            rows.append(dict(hub=net.hubs[h], disposition=d, actual=real.vol[h, j],
                             predicted_expected=res["est_pred"].vol[h, j], prior_based=res["est_prior"].vol[h, j]))
    vol_cmp = pd.DataFrame(rows)
    vol_cmp.to_csv(out / "volume_forecast_vs_actual.csv", index=False)
    metrics["volume_forecast_wape_hub_x_disposition"] = {
        "classifier_expected": wape(vol_cmp.predicted_expected, vol_cmp.actual),
        "prior_based": wape(vol_cmp.prior_based, vol_cmp.actual)}
    json.dump(metrics, open(out / "classifier_metrics.json", "w"), indent=2)

    # ---- per-order scoring file ----
    q = disposition_probs(week_df)
    scored = week_df[["order_id", "week_start", "hub", "customer_state", "category", "category_group", "price",
                      "weight_kg", "delay_days", "late", "review_score", "return_candidate"]].copy()
    scored["p_return"] = p_cal
    for j, d in enumerate(DISPOSITIONS):
        scored[f"assumed_p_{d}"] = q[:, j]
    ret = res["ret"].set_index("order_id")["disposition_idx"]
    scored["simulated_disposition"] = scored["order_id"].map(ret).map(lambda i: DISPOSITIONS[int(i)] if pd.notna(i) else "")
    scored.to_csv(out / "returns_scored.csv", index=False)

    # ---- scenario outputs (same schema the dashboard already reads) ----
    summaries = pd.DataFrame(res["summaries"])
    summaries.to_csv(out / "scenario_summary.csv", index=False)
    pd.concat(res["facs"]).to_csv(out / "facility_kpis.csv", index=False)
    pd.concat(res["routes"]).to_csv(out / "routing_plan.csv", index=False)
    pd.concat(res["lanes"]).to_csv(out / "realized_routing.csv", index=False)
    pd.DataFrame({"scenario": list(SCEN_LABELS), "scenario_label": list(SCEN_LABELS.values()),
                  "sort_order": [1, 2, 3, 4]}).to_csv(out / "dim_scenario.csv", index=False)
    for s in res["summaries"]:
        print(f"{s['scenario']:20s} open={s['facilities_open']} net=R${s['net_contribution']:>10,.0f} "
              f"recovered=R${s['recovered_value']:>10,.0f} backlog={s['backlog_units']:6.1f} "
              f"disposal={s['units_to_disposal']:6.1f} sla_breach={s['sla_breach_units']:6.1f} co2={s['co2_kg']:,.0f}kg")

    # ---- carbon shadow-price sweep (marginal abatement cost curve) for plan C ----
    sens = []
    for cp in CARBON_PRICES:
        e = replace(econ, carbon_price=cp)
        plan = solve_milp(net, res["est_pred"], e, f"C@{cp}")
        s, _, _, _ = evaluate(plan, res["est_pred"], real, net, e)
        sens.append(dict(carbon_price_brl_per_kg=cp, facilities_open=s["facilities_open"],
                         operating_cost_ex_carbon=s["operating_cost_ex_carbon"], co2_kg=s["co2_kg"],
                         co2_kg_per_unit=s["co2_kg_per_unit"], landfill_diversion_rate=s["landfill_diversion_rate"],
                         value_recovery_rate=s["value_recovery_rate"]))
    pd.DataFrame(sens).to_csv(out / "carbon_sensitivity.csv", index=False)

    # ---- network description ----
    pd.DataFrame({"facility": net.facilities, "weekly_capacity": net.cap_total, "fixed_cost_weekly": net.fixed_cost,
                  **{f"cap_{d}": net.cap_d[:, j] for j, d in enumerate(DISPOSITIONS)}}
                 ).to_csv(out / "network_facilities.csv", index=False)
    pd.DataFrame(net.dist_km, index=net.hubs, columns=net.facilities).round(1).to_csv(out / "network_distance_km.csv")

    # ---- weekly history for context ----
    wk = df.groupby("week_start").agg(delivered=("order_id", "size"), return_candidates=("return_candidate", "sum"),
                                      late_rate=("late", "mean"))
    wk["return_rate"] = wk.return_candidates / wk.delivered
    wk[wk.delivered >= 200].to_csv(out / "weekly_series.csv")

    cfg = dict(data_source="Olist Brazilian E-Commerce Public Dataset (Kaggle, olistbr/brazilian-ecommerce)",
               currency="BRL", return_definition=f"delivered order with review score <= {RETURN_SCORE_MAX} (PROXY)",
               seed=seed, planning_week_start=str(week.date()), planning_week_end=str((week + pd.Timedelta(days=6)).date()),
               n_orders_total=int(len(df)), n_train_orders=int(len(train_df)), n_week_delivered=int(len(week_df)),
               n_week_return_candidates=int(y.sum()), reference_weekly_volume=prep.ref_volume,
               week_vs_reference=float(y.sum() / prep.ref_volume), calibration=prep.cal, econ=econ.__dict__)
    json.dump(cfg, open(out / "run_config.json", "w"), indent=2, default=str)
    return summaries, metrics, cfg


def run_backtest(df, out: Path, seed: int, retrain_every: int = 4):
    weeks = eligible_weeks(df)
    econ0 = Econ()
    rows, prep = [], None
    for i, wk in enumerate(weeks):
        if i % retrain_every == 0:                      # retrain + recalibrate every 4 weeks
            prep = prepare(df, wk, econ0, seed, fast=True)
        res = plan_week(df, wk, prep, seed + i)
        week_df, y = res["week_df"], res["week_df"][clf.TARGET].values
        bl = clf.baselines(prep.train_df, week_df)
        m = clf.evaluate(res["p"], week_df, prep.train_df)
        base = dict(week=str(wk.date()), n_delivered=len(week_df), actual_returns=int(y.sum()),
                    reference_volume=prep.ref_volume, surge_ratio=float(y.sum() / prep.ref_volume),
                    auc=m["model"]["roc_auc"], brier=m["model"]["brier"], brier_hub_rate=m["hub_rate"]["brier"],
                    expected_classifier=float(res["p"].sum()), expected_hub_rate=float(bl["hub_rate"].sum()))
        for s in res["summaries"]:
            rows.append({**base, "scenario": s["scenario"], "net_contribution": s["net_contribution"],
                         "cost_to_serve": s["cost_to_serve"], "recovered_value": s["recovered_value"],
                         "backlog_units": s["backlog_units"], "units_to_disposal": s["units_to_disposal"],
                         "sla_breach_units": s["sla_breach_units"], "facilities_open": s["facilities_open"]})
        print(f"  backtest {wk.date()}  returns={int(y.sum()):4d}  ref={prep.ref_volume:5.0f}  auc={base['auc']:.3f}")
    bt = pd.DataFrame(rows)
    bt.to_csv(out / "backtest_weekly.csv", index=False)

    wide = bt.pivot(index="week", columns="scenario", values="net_contribution")
    info = bt.drop_duplicates("week").set_index("week")
    gain = pd.DataFrame({"C_minus_B": wide.C_milp_classifier - wide.B_milp_prior_only,
                         "C_minus_A": wide.C_milp_classifier - wide.A_greedy_nearest,
                         "D_minus_C": wide.D_milp_oracle - wide.C_milp_classifier,
                         "surge_ratio": info.surge_ratio, "actual_returns": info.actual_returns})
    gain.to_csv(out / "backtest_gain_by_week.csv")
    surge = gain.surge_ratio >= 1.3
    summary = dict(
        n_weeks=int(len(gain)), retrain_every_weeks=retrain_every,
        mean_net_contribution_by_plan={k: float(v) for k, v in wide.mean().items()},
        mean_gain_C_over_B=float(gain.C_minus_B.mean()), median_gain_C_over_B=float(gain.C_minus_B.median()),
        mean_gain_C_over_A=float(gain.C_minus_A.mean()), mean_gap_D_over_C=float(gain.D_minus_C.mean()),
        share_weeks_C_beats_B=float((gain.C_minus_B > 0).mean()), share_weeks_C_beats_A=float((gain.C_minus_A > 0).mean()),
        n_surge_weeks=int(surge.sum()),
        mean_gain_C_over_B_surge_weeks=float(gain.C_minus_B[surge].mean()) if surge.any() else None,
        mean_gain_C_over_B_normal_weeks=float(gain.C_minus_B[~surge].mean()),
        mean_auc=float(info.auc.mean()), mean_brier_model=float(info.brier.mean()),
        mean_brier_hub_rate=float(info.brier_hub_rate.mean()),
        returns_forecast_wape_classifier=float((info.expected_classifier - info.actual_returns).abs().sum() / info.actual_returns.sum()),
        returns_forecast_wape_hub_rate=float((info.expected_hub_rate - info.actual_returns).abs().sum() / info.actual_returns.sum()))
    json.dump(summary, open(out / "backtest_summary.json", "w"), indent=2)
    return summary


def main(data_dir: Path, week, out: Path, seed: int, backtest: bool, backtest_only: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    print("loading Olist and building the order table ...")
    df = build_order_table(load_raw(data_dir))
    wk = pd.Timestamp(week) if week else peak_week(df)
    print(f"orders={len(df):,}  return rate={df[clf.TARGET].mean():.3f}  featured planning week={wk.date()}")
    if not backtest_only:
        summaries, metrics, cfg = run_featured(df, wk, out, seed)
        c = metrics["calibrated"]
        print(f"\nclassifier AUC={c['model']['roc_auc']:.3f} brier={c['model']['brier']:.4f} "
              f"(hub-rate baseline brier={c['hub_rate']['brier']:.4f}) | returns: actual={metrics['expected_returns']['actual']} "
              f"classifier={metrics['expected_returns']['classifier']:.0f} hub-rate={metrics['expected_returns']['hub_rate']:.0f}")
    if backtest:
        print("\nrolling backtest ...")
        s = run_backtest(df, out, seed)
        print(json.dumps(s, indent=2))
    print(f"\nWrote outputs to {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/olist"))
    ap.add_argument("--week", type=str, default=None, help="Monday of the planning week (default: peak week)")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--backtest-only", action="store_true", help="run only the rolling backtest")
    a = ap.parse_args()
    main(a.data_dir, a.week, a.out, a.seed, backtest=not a.no_backtest or a.backtest_only,
         backtest_only=a.backtest_only)
