"""
Planning core: everything needed to plan and evaluate ONE week. Used by run_pipeline.py for the
featured week and for the rolling backtest.

No look-ahead: a model, calibration, hub rate or reference volume may only use orders whose
review had been ANSWERED before the planning week starts.
"""
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from . import classifier as clf
from . import demand as dm
from .calibration import apply_calibration, calibrate_logistics
from .config import Econ, HUBS
from .evaluation import evaluate, realized_lane_table, routing_table
from .network import build_network
from .optimizer import greedy_nearest, solve_milp

MIN_WEEK_ORDERS = 1000          # a week must have at least this many deliveries to be a planning week
REF_WEEKS = 13                  # reference volume = median weekly returns over the last 13 weeks


def split_week(df: pd.DataFrame, week_start) -> tuple:
    """(train_df, week_df). Training uses only orders delivered AND with their review answered
    before the week starts, i.e. labels that would really be known at planning time."""
    w = pd.Timestamp(week_start)
    train = df[(df["order_delivered_customer_date"] < w) & (df["review_answer_timestamp"] < w)]
    week = df[df["week_start"] == w]
    return train.reset_index(drop=True), week.reset_index(drop=True)


def reference_volume(train_df: pd.DataFrame, week_start) -> float:
    """Typical weekly return volume the network was sized for: median over the last REF_WEEKS
    complete weeks before the planning week."""
    w = pd.Timestamp(week_start)
    t = train_df[train_df["week_start"] >= w - pd.Timedelta(weeks=REF_WEEKS)]
    wk = t.groupby("week_start").agg(n=("order_id", "size"), r=("return_candidate", "sum"))
    wk = wk[wk.n >= 500]
    return float(wk.r.median()) if len(wk) else float(train_df["return_candidate"].sum() / 20)


def eligible_weeks(df: pd.DataFrame, start="2018-01-08", end="2018-08-27") -> list:
    wk = df.groupby("week_start").agg(n=("order_id", "size"), r=("return_candidate", "sum"))
    wk = wk[(wk.n >= MIN_WEEK_ORDERS) & (wk.index >= pd.Timestamp(start)) & (wk.index <= pd.Timestamp(end))]
    return list(wk.index)


def peak_week(df: pd.DataFrame, **kw):
    """The eligible week with the most return candidates (a real disruption week)."""
    wk = df[df["week_start"].isin(eligible_weeks(df, **kw))].groupby("week_start")["return_candidate"].sum()
    return wk.idxmax()


@dataclass
class Prepared:
    """Everything that depends only on the training window (reused across weeks in a backtest block)."""
    train_df: pd.DataFrame
    econ: Econ
    cal: dict
    raw_model: object
    cal_model: object
    ref_volume: float


def prepare(df: pd.DataFrame, as_of, econ0: Econ, seed: int, fast: bool) -> Prepared:
    train_df, _ = split_week(df, as_of)
    cal = calibrate_logistics(train_df, seed)
    econ = apply_calibration(econ0, cal)
    raw, calm = clf.train(train_df, seed, fast)
    return Prepared(train_df, econ, cal, raw, calm, reference_volume(train_df, as_of))


def plan_week(df: pd.DataFrame, week_start, prep: Prepared, seed: int) -> dict:
    """Plan and evaluate one week with the four policies. `prep` may come from an earlier date
    (backtest blocks); only its training data is used, never this week's labels."""
    w = pd.Timestamp(week_start)
    week_df = df[df["week_start"] == w].reset_index(drop=True)
    train_df, econ = prep.train_df, prep.econ
    H = len(HUBS)

    p = clf.predict(prep.cal_model, week_df)
    real, ret = dm.realized_demand(week_df, H, seed, train_df)
    est_pred = dm.expected_demand(week_df, p, H, train_df)
    est_prior = dm.prior_demand(train_df, week_df, H)
    net = build_network(prep.ref_volume, econ)

    scenarios = [
        ("A_greedy_nearest",  greedy_nearest(net, est_pred, econ, "A_greedy_nearest"), est_pred),
        ("B_milp_prior_only", solve_milp(net, est_prior, econ, "B_milp_prior_only"), est_prior),
        ("C_milp_classifier", solve_milp(net, est_pred, econ, "C_milp_classifier"), est_pred),
        # perfect-information lower bound: true volumes AND no planning buffer, i.e. exactly the
        # constraints of the realized world
        ("D_milp_oracle",     solve_milp(net, real, replace(econ, cap_buffer=0.0), "D_milp_oracle"), real),
    ]
    out = dict(week_df=week_df, train_df=train_df, p=p, real=real, ret=ret, est_pred=est_pred,
               est_prior=est_prior, net=net, econ=econ, scenarios=scenarios, summaries=[], facs=[],
               routes=[], lanes=[])
    for name, plan, plan_dem in scenarios:
        s, f, flows_real, _ = evaluate(plan, plan_dem, real, net, econ)
        out["summaries"].append(s)
        out["facs"].append(f)
        out["routes"].append(routing_table(plan, net, econ))
        out["lanes"].append(realized_lane_table(plan, flows_real, real, net, econ))
    return out
