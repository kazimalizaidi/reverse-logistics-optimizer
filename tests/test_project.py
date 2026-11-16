"""
Run:  pytest -q

Groups
  1. MILP checked against instances solvable by hand (including the calibrated shipping formula)
  2. Olist data layer on a tiny fake dataset (label rule, filters, weeks, hubs, dispositions)
  3. No look-ahead in the train / planning-week split
  4. Calibration recovers a known formula; demand builders conserve units
  5. Plan invariants and evaluator/solver consistency on a synthetic order table
  6. Optional smoke test on the real Olist files (skipped when data/olist/ is absent)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl_network import classifier as clf
from rl_network import demand as dm
from rl_network import olist_data as od
from rl_network import planning as pl
from rl_network.calibration import calibrate_logistics
from rl_network.config import DISPOSITION_MIX, DISPOSITIONS, HUBS, STATE_TO_HUB, Econ
from rl_network.demand import Demand
from rl_network.evaluation import evaluate
from rl_network.network import Network, build_network
from rl_network.optimizer import greedy_nearest, landfill_unit_cost, solve_milp

TOL = 1e-4
ROOT = Path(__file__).resolve().parents[1]


# ============================================================================ 1. MILP by hand
def tiny_econ(**kw):
    base = dict(ship_cost_per_kg_km=0.01, carbon_price=0.0, late_penalty=0.0, min_util=0.0, cap_buffer=0.0)
    base.update(kw)
    return Econ(**base)


def tiny_net(caps, dists, fixed=(0, 0)):
    """1 hub, 2 facilities, restock only; zero processing cost, perfect yield."""
    F = 2
    cap_d = np.zeros((F, 3)); cap_d[:, 0] = caps
    return Network(["H"], ["A", "B"], np.array([dists], dtype=float), np.array([[0.5, 0.5]]), cap_d,
                   np.array(caps, dtype=float), np.array(fixed, dtype=float), np.zeros((F, 3)), np.ones((F, 3)))


def tiny_demand(units):
    vol = np.zeros((1, 3)); vol[0, 0] = units
    return Demand(vol, np.zeros((1, 3)), np.ones((1, 3)))      # price 0, weight 1 kg


def test_fills_cheapest_first_hand_solved():
    # A: 100 km cap 50 -> 1/unit ; B: 300 km cap 100 -> 3/unit ; demand 80  =>  50*1 + 30*3 = 140
    plan = solve_milp(tiny_net([50, 100], [100, 300]), tiny_demand(80), tiny_econ())
    assert plan.status == "Optimal"
    assert plan.flows[0, 0, 0] == pytest.approx(50, abs=TOL)
    assert plan.flows[0, 1, 0] == pytest.approx(30, abs=TOL)
    assert plan.objective == pytest.approx(140, abs=TOL)


def test_calibrated_shipping_formula_hand_solved():
    # parcel cost = base + per_km*km + per_kg_km*kg*km ; base 5, 0.1/km, no weight term
    # A: 5+10 = 15 ; B: 5+30 = 35 ; fill A (50) then B (30) => 50*15 + 30*35 = 1800
    # (disposal made expensive so it is not the cheaper option: at the default fee of 8 per unit,
    #  disposing of everything, 80 * 8 = 640, would beat shipping at 15 or 35)
    econ = tiny_econ(ship_base_cost=5.0, ship_cost_per_km=0.1, ship_cost_per_kg_km=0.0, landfill_fee=1000.0)
    plan = solve_milp(tiny_net([50, 100], [100, 300]), tiny_demand(80), econ)
    assert plan.objective == pytest.approx(1800, abs=TOL)


def test_overflow_goes_to_disposal():
    econ = tiny_econ()
    plan = solve_milp(tiny_net([50, 100], [100, 300]), tiny_demand(200), econ)
    assert plan.landfill[0, 0] == pytest.approx(50, abs=TOL)
    assert plan.objective == pytest.approx(50 * 1 + 100 * 3 + 50 * landfill_unit_cost(econ), abs=TOL)


def test_closes_expensive_facility():
    plan = solve_milp(tiny_net([100, 100], [100, 100], fixed=(0, 1000)), tiny_demand(60), tiny_econ())
    assert plan.open_[1] == 0 and plan.flows[0, 1, 0] == pytest.approx(0, abs=TOL)
    assert plan.objective == pytest.approx(60.0, abs=TOL)


def test_capacity_buffer_is_respected():
    plan = solve_milp(tiny_net([100, 100], [100, 300]), tiny_demand(95), tiny_econ(cap_buffer=0.10))
    assert plan.flows[0, 0, 0] <= 90 + TOL


# ============================================================================ 2. Olist data layer
def fake_raw():
    """A 6-order fake Olist dataset with hand-checkable properties."""
    t0 = pd.Timestamp("2018-03-05 10:00")                  # a Monday
    d = lambda days: t0 + pd.Timedelta(days=days)
    orders = pd.DataFrame({
        "order_id": list("abcdef"), "customer_id": [f"c{i}" for i in range(6)],
        "order_status": ["delivered"] * 4 + ["canceled", "delivered"],
        "order_purchase_timestamp": [t0] * 6,
        "order_approved_at": [d(0.1)] * 6,
        "order_delivered_carrier_date": [d(1), d(1), d(1), d(1), pd.NaT, d(1)],
        "order_delivered_customer_date": [d(6), d(14), d(9), d(8), pd.NaT, d(7)],     # a late, b/c on time
        "order_estimated_delivery_date": [d(10), d(10), d(10), d(10), d(10), d(10)],
    })
    items = pd.DataFrame({"order_id": list("abcdef") + ["a"], "order_item_id": [1] * 6 + [2],
                          "product_id": ["p1", "p2", "p1", "p2", "p1", "p3", "p2"], "seller_id": ["s1"] * 7,
                          "price": [100.0, 50.0, 80.0, 30.0, 10.0, 60.0, 20.0],
                          "freight_value": [15.0, 20.0, 12.0, 10.0, 9.0, 11.0, 5.0]})
    reviews = pd.DataFrame({"order_id": list("abcde"), "review_score": [1, 2, 3, 5, 1],
                            "review_creation_date": [d(7)] * 5,
                            "review_answer_timestamp": [d(8)] * 5})        # order f has no review
    customers = pd.DataFrame({"customer_id": [f"c{i}" for i in range(6)],
                              "customer_state": ["SP", "RJ", "BA", "PE", "SP", "SP"],
                              "customer_zip_code_prefix": [1000, 2000, 3000, 4000, 1000, 1000]})
    products = pd.DataFrame({"product_id": ["p1", "p2", "p3"],
                             "product_category_name": ["informatica_acessorios", "moveis_decoracao", None],
                             "product_weight_g": [500.0, 2000.0, np.nan], "product_length_cm": [10.0, 30.0, np.nan],
                             "product_height_cm": [5.0, 20.0, np.nan], "product_width_cm": [10.0, 25.0, np.nan]})
    sellers = pd.DataFrame({"seller_id": ["s1"], "seller_state": ["SP"], "seller_zip_code_prefix": [1000]})
    geo = pd.DataFrame({"geolocation_zip_code_prefix": [1000, 2000, 3000, 4000],
                        "geolocation_lat": [-23.5, -22.9, -12.9, -8.0], "geolocation_lng": [-46.6, -43.2, -38.5, -34.9]})
    trans = pd.DataFrame({"product_category_name": ["informatica_acessorios", "moveis_decoracao"],
                          "product_category_name_english": ["computers_accessories", "furniture_decor"]})
    return dict(orders=orders, items=items, reviews=reviews, customers=customers, products=products,
                sellers=sellers, geo=geo, translation=trans)


def test_order_table_filters_and_label_rule():
    df = od.build_order_table(fake_raw()).set_index("order_id")
    assert set(df.index) == {"a", "b", "c", "d"}                  # canceled (e) and unreviewed (f) removed
    assert df.loc["a", "return_candidate"] == 1 and df.loc["b", "return_candidate"] == 1   # scores 1 and 2
    assert df.loc["c", "return_candidate"] == 0 and df.loc["d", "return_candidate"] == 0   # scores 3 and 5


def test_order_table_features_are_correct():
    df = od.build_order_table(fake_raw()).set_index("order_id")
    assert df.loc["a", "n_items"] == 2 and df.loc["a", "price"] == pytest.approx(120.0)   # two items summed
    assert df.loc["b", "late"] == 1 and df.loc["b", "delay_days"] == pytest.approx(4.0)   # 14 days vs 10 promised
    assert df.loc["a", "late"] == 0 and df.loc["a", "delay_days"] == pytest.approx(-4.0)
    assert df.loc["a", "same_state"] == 1 and df.loc["b", "same_state"] == 0
    assert df.loc["a", "seller_customer_km"] < 50 and df.loc["d", "seller_customer_km"] > 1000
    assert df.loc["a", "hub"] == "São Paulo" and df.loc["b", "hub"] == "Rio de Janeiro"
    assert df.loc["c", "hub"] == "Salvador" and df.loc["d", "hub"] == "Recife"
    assert df.loc["a", "category_group"] == "electronics" and df.loc["b", "category_group"] == "home"
    assert (df.weight_kg > 0).all()                                # missing weights were imputed


def test_week_start_is_monday_of_delivery():
    df = od.build_order_table(fake_raw())
    assert (df.week_start.dt.dayofweek == 0).all()
    assert (df.week_start <= df.order_delivered_customer_date).all()
    assert ((df.order_delivered_customer_date - df.week_start) < pd.Timedelta(days=7)).all()


def test_every_state_maps_to_a_known_hub():
    hubs = {h[0] for h in HUBS}
    assert set(STATE_TO_HUB.values()) <= hubs and hubs <= set(STATE_TO_HUB.values())
    assert len(STATE_TO_HUB) == 27                                # all 26 states + Distrito Federal


def test_disposition_mix_rows_sum_to_one():
    for g, row in DISPOSITION_MIX.items():
        assert sum(row) == pytest.approx(1.0) and min(row) >= 0, g


def test_disposition_probs_valid_and_late_shifts_to_restock():
    df = pd.DataFrame({"category_group": ["electronics", "electronics"], "late": [0, 1]})
    p = od.disposition_probs(df)
    assert np.allclose(p.sum(axis=1), 1.0) and (p >= 0).all()
    assert p[1, 0] > p[0, 0]                                      # late delivery -> more restock


def test_sampled_dispositions_follow_probabilities():
    df = pd.DataFrame({"category_group": ["fashion"] * 20000, "late": [0] * 20000})
    draw = od.sample_dispositions(df, seed=1)
    expected = np.array(DISPOSITION_MIX["fashion"])
    assert np.allclose(np.bincount(draw, minlength=3) / len(draw), expected, atol=0.02)
    assert np.array_equal(draw, od.sample_dispositions(df, seed=1))      # reproducible


def test_model_features_never_include_the_label_or_review():
    assert not {"review_score", "return_candidate", "review_creation_date", "review_answer_timestamp"} & set(clf.FEATURES)


# ============================================================================ synthetic order table
def synth_orders(n=4500, seed=0):
    """Order table with the same columns as olist_data.build_order_table, with a known signal:
    late deliveries return much more often."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2018-01-01")
    deliv = start + pd.to_timedelta(rng.uniform(0, 140, n), unit="D")
    hub_idx = rng.integers(0, len(HUBS), n)
    km = rng.uniform(30, 2500, n); kg = rng.uniform(0.2, 8, n); price = rng.uniform(20, 400, n)
    delay = rng.normal(-5, 6, n)
    p = 1 / (1 + np.exp(-(-2.2 + 0.18 * delay)))
    ret = (rng.random(n) < p).astype(int)
    day = pd.Timedelta(days=1)
    df = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(n)],
        "order_purchase_timestamp": deliv - 10 * day,
        "order_delivered_customer_date": deliv,
        "review_creation_date": deliv + 1 * day,
        "review_answer_timestamp": deliv + 3 * day,
        "hub": [HUBS[i][0] for i in hub_idx], "hub_idx": hub_idx,
        "customer_state": "SP", "seller_state": "SP", "category": "x",
        "category_group": rng.choice(list(DISPOSITION_MIX), n),
        "price": price, "freight_value": 10 + 0.004 * km + 0.002 * kg * km + rng.normal(0, 1, n),
        "weight_kg": kg, "volume_cm3": rng.uniform(100, 5000, n), "n_items": rng.integers(1, 3, n),
        "freight_ratio": 0.1, "delivery_days": 10 + delay, "est_days": 15.0, "delay_days": delay,
        "handoff_days": 2.0, "transit_days": 1 + 0.3 * np.sqrt(km) + rng.normal(0, 0.5, n),
        "late": (delay > 0).astype(int), "seller_customer_km": km, "same_state": 1,
        "purchase_dow": rng.integers(0, 7, n), "review_score": np.where(ret == 1, 1, 5), "return_candidate": ret})
    d = df.order_delivered_customer_date.dt.normalize()
    df["week_start"] = d - pd.to_timedelta(d.dt.dayofweek, unit="D")
    return df.sort_values("order_delivered_customer_date").reset_index(drop=True)


@pytest.fixture(scope="module")
def synth():
    return synth_orders()


# ============================================================================ 3. no look-ahead
def test_split_has_no_lookahead(synth):
    w = pd.Timestamp("2018-04-23")
    train, week = pl.split_week(synth, w)
    assert (train.order_delivered_customer_date < w).all()
    assert (train.review_answer_timestamp < w).all()              # only labels that were already known
    assert (week.week_start == w).all() and len(week) > 0
    assert set(train.order_id).isdisjoint(set(week.order_id))


def test_reference_volume_uses_only_past_weeks(synth):
    w = pd.Timestamp("2018-04-23")
    train, _ = pl.split_week(synth, w)
    ref = pl.reference_volume(train, w)
    assert ref > 0
    future = synth[synth.week_start >= w]
    assert not set(future.order_id) & set(train.order_id)


# ============================================================================ 4. calibration, demand
def test_calibration_recovers_known_formula(synth):
    cal = calibrate_logistics(synth, seed=0, sample=3000)
    f, t = cal["freight"], cal["transit"]
    assert f["ship_base_cost"] == pytest.approx(10.0, rel=0.15)
    assert f["ship_cost_per_km"] == pytest.approx(0.004, rel=0.25)
    assert f["ship_cost_per_kg_km"] == pytest.approx(0.002, rel=0.25)
    assert t["transit_base_days"] == pytest.approx(1.0, abs=0.4)
    assert t["transit_sqrt_coef"] == pytest.approx(0.3, rel=0.1)


def test_demand_builders_conserve_units(synth):
    train, week = pl.split_week(synth, pd.Timestamp("2018-04-23"))
    H = len(HUBS)
    p = np.full(len(week), 0.25)
    est = dm.expected_demand(week, p, H, train)
    assert est.total == pytest.approx(p.sum(), rel=1e-9)
    real, ret = dm.realized_demand(week, H, seed=1, train_df=train)
    assert real.total == pytest.approx(week.return_candidate.sum()) and len(ret) == week.return_candidate.sum()
    prior = dm.prior_demand(train, week, H)
    expected_prior = sum((week.hub_idx == h).sum() * train[train.hub_idx == h].return_candidate.mean() for h in range(H))
    assert prior.total == pytest.approx(expected_prior, rel=1e-9)
    assert (est.avg_price > 0).all() and (real.avg_weight > 0).all() and (prior.avg_price > 0).all()


# ============================================================================ 5. plan invariants
@pytest.fixture(scope="module")
def planned(synth):
    w = pd.Timestamp("2018-04-23")
    prep = pl.prepare(synth, w, Econ(), seed=0, fast=True)
    return prep, pl.plan_week(synth, w, prep, seed=0)


def test_classifier_beats_the_global_rate_on_known_signal(planned):
    prep, res = planned
    m = clf.evaluate(res["p"], res["week_df"], prep.train_df)
    assert m["model"]["roc_auc"] > 0.65          # noisy synthetic signal, clearly above 0.5
    assert m["model"]["brier"] < m["global_rate"]["brier"]


def test_plan_constraints_hold(planned):
    prep, res = planned
    net, econ = res["net"], res["econ"]
    for name, plan, dem in res["scenarios"]:
        assert np.allclose(plan.flows.sum(axis=1) + plan.landfill, dem.vol, atol=1e-3), name      # conservation
        buffer = 0.0 if name.startswith("D_") else econ.cap_buffer
        assert (plan.flows.sum(axis=0) <= net.cap_d * (1 - buffer) + 1e-3).all(), name              # line capacity
        assert (plan.flows.sum(axis=(0, 2)) <= net.cap_total * (1 - buffer) + 1e-3).all(), name     # site capacity
        assert (plan.flows.sum(axis=(0, 2))[plan.open_ == 0] < 1e-6).all(), name                    # closed sites idle
        assert (plan.flows.sum(axis=0)[~net.supports] < 1e-9).all(), name                           # unsupported lines idle


def test_evaluator_conserves_units_and_flags_backlog(planned):
    _, res = planned
    for s in res["summaries"]:
        assert s["units_processed"] + s["units_to_disposal"] == pytest.approx(s["units_returned"], abs=1e-3)
        assert s["backlog_units"] >= 0 and 0 <= s["sla_breach_rate"] <= 1 + 1e-9
        assert s["net_contribution"] == pytest.approx(s["recovered_value"] - s["cost_to_serve"], rel=1e-9, abs=1e-6)


def test_evaluator_matches_solver_objective_under_perfect_information(synth):
    """With true volumes and no buffer nothing is rescaled, so the evaluator's realized net cost
    must equal the solver's objective."""
    train, week = pl.split_week(synth, pd.Timestamp("2018-04-23"))
    econ = Econ(cap_buffer=0.0)
    real, _ = dm.realized_demand(week, len(HUBS), 1, train)
    net = build_network(max(real.total, 50.0), econ)
    plan = solve_milp(net, real, econ)
    s, _, _, _ = evaluate(plan, real, real, net, econ)
    assert s["net_cost"] == pytest.approx(plan.objective, rel=1e-9, abs=1e-4)


def test_milp_never_worse_than_greedy_on_its_own_objective(planned):
    _, res = planned
    net, econ, est = res["net"], res["econ"], res["est_pred"]
    assert solve_milp(net, est, econ).objective <= greedy_nearest(net, est, econ).objective + 1e-6


# ============================================================================ 6. real data (optional)
REAL = ROOT / "data" / "olist" / "olist_orders_dataset.csv"


@pytest.mark.skipif(not REAL.exists(), reason="Olist CSVs not found in data/olist/")
def test_real_olist_table_sanity():
    df = od.build_order_table(od.load_raw(ROOT / "data" / "olist"))
    assert df.order_id.is_unique and len(df) > 50000
    assert 0.05 < df.return_candidate.mean() < 0.25
    assert df.late.mean() < 0.2
    assert set(df.hub) <= {h[0] for h in HUBS}
    assert df.week_start.dt.dayofweek.eq(0).all()
    assert df[clf.FEATURES].isna().mean().max() < 0.05
