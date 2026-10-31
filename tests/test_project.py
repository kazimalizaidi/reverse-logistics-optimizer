"""
Run:  pytest -q
The first group verifies the MILP against instances solvable by hand; the second checks
structural invariants on a full (small) end-to-end run without the classifier.
"""
import numpy as np
import pytest

from rl_network import demand as dm
from rl_network.config import DISPOSITIONS, Econ
from rl_network.data_generation import generate_returns
from rl_network.demand import Demand
from rl_network.evaluation import evaluate
from rl_network.network import Network, build_network
from rl_network.optimizer import greedy_nearest, landfill_unit_cost, solve_milp

TOL = 1e-4


def tiny_econ(**kw):
    base = dict(ship_cost_per_kg_km=0.01, carbon_price=0.0, late_penalty=0.0,
                min_util=0.0, cap_buffer=0.0)
    base.update(kw)
    return Econ(**base)


def tiny_net(caps, dists, fixed=(0, 0)):
    """1 hub, 2 facilities, restock only; zero processing cost, perfect yield."""
    F = 2
    cap_d = np.zeros((F, 3)); cap_d[:, 0] = caps
    cap_total = np.array(caps, dtype=float)
    return Network(["H"], ["A", "B"], np.array([dists], dtype=float),
                   np.array([[0.5, 0.5]]), cap_d, cap_total, np.array(fixed, dtype=float),
                   np.zeros((F, 3)), np.ones((F, 3)))


def tiny_demand(units):
    vol = np.zeros((1, 3)); vol[0, 0] = units
    return Demand(vol, np.zeros((1, 3)), np.ones((1, 3)))   # price 0, weight 1 kg


def test_fills_cheapest_first_hand_solved():
    # A: 100 km cap 50 -> $1/unit ; B: 300 km cap 100 -> $3/unit ; demand 80
    # optimum = 50*1 + 30*3 = 140
    plan = solve_milp(tiny_net([50, 100], [100, 300]), tiny_demand(80), tiny_econ())
    assert plan.status == "Optimal"
    assert plan.flows[0, 0, 0] == pytest.approx(50, abs=TOL)
    assert plan.flows[0, 1, 0] == pytest.approx(30, abs=TOL)
    assert plan.objective == pytest.approx(140, abs=TOL)


def test_overflow_goes_to_disposal():
    # capacity 150 total, demand 200 -> 50 units to disposal at the landfill unit cost
    econ = tiny_econ()
    plan = solve_milp(tiny_net([50, 100], [100, 300]), tiny_demand(200), econ)
    assert plan.landfill[0, 0] == pytest.approx(50, abs=TOL)
    expected = 50 * 1 + 100 * 3 + 50 * landfill_unit_cost(econ)
    assert plan.objective == pytest.approx(expected, abs=TOL)


def test_closes_expensive_facility():
    # A alone can absorb the demand; B has a large fixed cost -> must stay closed
    plan = solve_milp(tiny_net([100, 100], [100, 100], fixed=(0, 1000)), tiny_demand(60), tiny_econ())
    assert plan.open_[1] == 0 and plan.flows[0, 1, 0] == pytest.approx(0, abs=TOL)
    assert plan.objective == pytest.approx(60 * 1.0, abs=TOL)


def test_capacity_buffer_is_respected():
    plan = solve_milp(tiny_net([100, 100], [100, 300]), tiny_demand(95), tiny_econ(cap_buffer=0.10))
    assert plan.flows[0, 0, 0] <= 90 + TOL          # 10% headroom on A


@pytest.fixture(scope="module")
def world():
    econ = Econ()
    wk = generate_returns(3000, 7)
    net = build_network(len(wk), econ)
    real = dm.from_labels(wk, net.H)
    return econ, wk, net, real


def test_generator_probabilities_valid():
    df = generate_returns(5000, 1)
    tp = df[[f"_true_p_{d}" for d in DISPOSITIONS]].values
    assert np.allclose(tp.sum(axis=1), 1.0)
    assert (df["price"] > 0).all() and (df["weight_kg"] > 0).all()
    # sampled label frequencies match the mean true probabilities (sampling noise only)
    freq = np.bincount(df["disposition_idx"], minlength=3) / len(df)
    assert np.allclose(freq, tp.mean(axis=0), atol=0.02)


def test_model_features_exclude_labels_and_true_probs():
    from rl_network.classifier import FEATURES
    assert not any(f.startswith("_true_p_") or f.startswith("disposition") for f in FEATURES)


def test_plan_constraints_hold(world):
    econ, wk, net, real = world
    for plan in (solve_milp(net, real, econ), greedy_nearest(net, real, econ)):
        # (1) flow conservation
        assert np.allclose(plan.flows.sum(axis=1) + plan.landfill, real.vol, atol=1e-3)
        # (2)/(3) capacities with buffer, closed sites untouched
        load_fd = plan.flows.sum(axis=0)
        assert (load_fd <= net.cap_d * (1 - econ.cap_buffer) + 1e-3).all()
        assert (plan.flows.sum(axis=(0, 2)) <= net.cap_total * (1 - econ.cap_buffer) + 1e-3).all()
        assert (plan.flows.sum(axis=(0, 2))[plan.open_ == 0] < 1e-6).all()
        # unsupported (facility, disposition) pairs carry nothing
        assert (plan.flows.sum(axis=0)[~net.supports] < 1e-9).all()


def test_evaluator_matches_solver_objective(world):
    """With perfect information and no buffer nothing is rescaled, so the evaluator's
    realized net cost must equal the solver's objective to rounding."""
    econ, wk, net, real = world
    e0 = Econ(cap_buffer=0.0)
    plan = solve_milp(net, real, e0)
    s, _, _, _ = evaluate(plan, real, real, net, e0)
    assert s["net_cost"] == pytest.approx(plan.objective, rel=1e-9, abs=1e-4)


def test_milp_never_worse_than_greedy_on_its_own_objective(world):
    econ, wk, net, real = world
    assert solve_milp(net, real, econ).objective <= greedy_nearest(net, real, econ).objective + 1e-6


def test_realized_units_are_conserved(world):
    econ, wk, net, real = world
    plan = solve_milp(net, real, econ)
    s, fac, flows, landfill = evaluate(plan, real, real, net, econ)
    assert flows.sum() + landfill.sum() == pytest.approx(real.vol.sum(), abs=1e-3)
    assert s["units_processed"] + s["units_to_disposal"] == pytest.approx(s["units_returned"], abs=1e-3)


def test_dashboard_exports_reconcile_to_scenario_totals(world):
    """Lane-level and facility-level tables feeding Power BI must add up to the scenario summary."""
    from rl_network.evaluation import realized_lane_table
    econ, wk, net, real = world
    for plan in (solve_milp(net, real, econ), greedy_nearest(net, real, econ)):
        s, fac, flows, landfill = evaluate(plan, real, real, net, econ)
        lanes = realized_lane_table(plan, flows, real, net, econ)
        assert lanes["units"].sum() == pytest.approx(s["units_processed"], abs=1e-6)
        assert lanes["recovered_value"].sum() == pytest.approx(s["recovered_value"], rel=1e-9)
        assert lanes["sla_breach_units"].sum() == pytest.approx(s["sla_breach_units"], abs=1e-6)
        assert fac["backlog_units"].sum() == pytest.approx(s["backlog_units"], abs=1e-6)
        assert fac["cost_to_serve"].sum() + s["disposal_cost"] == pytest.approx(s["cost_to_serve"], rel=1e-9)
