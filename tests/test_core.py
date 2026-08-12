import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from hjb_fdm import evaluate_solution_at_x0, interpolate_at_y, pi_from_grid, solve_hjb_fdm, terminal_condition  # noqa: E402
from merton_closed_form import MertonModel  # noqa: E402

from exp_merton_aapl_empirical import build_monthly_estimates, compute_metrics, run_backtest  # noqa: E402
from exp_hedging_transaction_costs import (  # noqa: E402
    bs_delta,
    call_payoff,
    metrics_from_result as hedge_metrics_from_result,
    simulate_gbm_paths,
    simulate_hedge,
    transaction_cost,
)
import exp_hedging_bellman as bellman  # noqa: E402
import exp_barrier_hedging_bellman as barrier  # noqa: E402


def assert_close(value, target, tol, name):
    if not abs(value - target) <= tol:
        raise AssertionError(f"{name}: {value} not within {tol} of {target}")


def test_merton_identity():
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    x = np.array([0.5, 1.0, 2.0])
    t = 0.3
    pi_from_derivatives = -(model.mu - model.r) / model.sigma ** 2 * model.Vx(t, x) / model.Vxx(t, x)
    np.testing.assert_allclose(pi_from_derivatives, model.pi_amount(x), rtol=1e-12, atol=1e-12)


def test_terminal_condition_is_crra_utility():
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    y = np.linspace(-1.0, 1.0, 11)
    x = np.exp(y)
    np.testing.assert_allclose(terminal_condition(model, y), x ** (1.0 - model.gamma) / (1.0 - model.gamma))


def test_analytic_value_formula_matches_requested_benchmark():
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=5.0, T=1.0)
    t = 0.25
    x = np.array([0.7, 1.0, 1.6])
    exponent = (1.0 - model.gamma) * (
        model.r + (model.mu - model.r) ** 2 / (2.0 * model.gamma * model.sigma ** 2)
    ) * (model.T - t)
    expected = x ** (1.0 - model.gamma) / (1.0 - model.gamma) * np.exp(exponent)
    np.testing.assert_allclose(model.V(t, x), expected, rtol=1e-12, atol=1e-12)


def test_gamma_3_analytic_control_is_half_x():
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    x = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(model.pi_amount(x), 0.5 * x, rtol=1e-12, atol=1e-12)


def test_log_transform_vxx():
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    y = np.linspace(-1.0, 1.0, 2001)
    dy = y[1] - y[0]
    t = 0.4
    w = model.V(t, np.exp(y))
    w_y = np.gradient(w, dy)
    w_yy = np.gradient(w_y, dy)
    Vxx_from_y = (w_yy - w_y) * np.exp(-2 * y)
    Vxx_exact = model.Vxx(t, np.exp(y))
    np.testing.assert_allclose(Vxx_from_y[10:-10], Vxx_exact[10:-10], rtol=2e-4, atol=2e-4)


def test_interpolation_at_y0_matches_odd_node():
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    y, w = solve_hjb_fdm(model, Ny=101, Nt=400)
    ev = evaluate_solution_at_x0(model, y, w, x0=1.0)
    assert_close(ev["nearest_y"], 0.0, 1e-14, "nearest_y")
    assert_close(ev["V_nearest"], ev["V_interp"], 1e-14, "V interpolation at node")
    assert_close(ev["pi_nearest"], ev["pi_interp"], 1e-14, "pi interpolation at node")


def test_interpolation_at_x0_between_nodes_is_linear():
    y = np.linspace(-3.0, 3.0, 50)
    values = 2.0 * y + 1.0
    assert_close(interpolate_at_y(y, values, np.log(1.0)), 1.0, 1e-14, "linear interpolation")


def test_hjb_solution_has_no_nan_or_inf():
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    y, w, stats = solve_hjb_fdm(model, Ny=100, Nt=100, return_diagnostics=True)
    pi = pi_from_grid(model, y, w)
    if not np.all(np.isfinite(w)):
        raise AssertionError("HJB value grid contains NaN or Inf")
    if not np.all(np.isfinite(pi)):
        raise AssertionError("HJB control grid contains NaN or Inf")
    if stats["has_nan"] or stats["has_inf"]:
        raise AssertionError("HJB diagnostics reported NaN or Inf")


def synthetic_empirical_data(n_days=760):
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    log_returns = 0.0004 + 0.01 * np.sin(np.arange(n_days) / 17.0)
    adj_close = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame(
        {
            "date": dates,
            "adj_close": adj_close,
            "stock_simple_return": np.exp(log_returns) - 1.0,
            "stock_log_return": log_returns,
            "rf_annual": 0.02,
            "rf_daily_return": np.exp(0.02 / 252.0) - 1.0,
        }
    )


def test_empirical_no_future_information_used():
    data = synthetic_empirical_data()
    estimates = build_monthly_estimates(data, window=504, gammas=[3.0])
    if not (estimates["estimation_end"] < estimates["decision_date"]).all():
        raise AssertionError("A decision used returns that were not strictly before the decision date")
    if not estimates["window_observations"].eq(504).all():
        raise AssertionError("A decision did not use exactly 504 observations")


def test_empirical_first_backtest_date_after_504_observations():
    data = synthetic_empirical_data()
    estimates = build_monthly_estimates(data, window=504, gammas=[3.0])
    first_idx = int(estimates["decision_idx"].min())
    if first_idx < 504:
        raise AssertionError("Backtest starts before 504 observations are available")


def test_empirical_long_only_weights_are_bounded():
    data = synthetic_empirical_data()
    estimates = build_monthly_estimates(data, window=504, gammas=[1.5, 3.0, 5.0])
    if not ((estimates["w_long_only"] >= 0.0) & (estimates["w_long_only"] <= 1.0)).all():
        raise AssertionError("Long-only Merton weights left [0, 1]")


def test_empirical_cash_benchmark_follows_risk_free_rate():
    data = synthetic_empirical_data()
    estimates = build_monthly_estimates(data, window=504, gammas=[3.0])
    daily, _ = run_backtest(data, estimates)
    cash = daily[daily["strategy"].eq("benchmark_cash_100")].sort_values("date")
    expected = np.cumprod(1.0 + data.loc[int(estimates["decision_idx"].min()):, "rf_daily_return"].to_numpy())
    np.testing.assert_allclose(cash["wealth"].to_numpy(), expected, rtol=1e-12, atol=1e-12)


def test_empirical_no_nan_inf_in_traded_periods():
    data = synthetic_empirical_data()
    estimates = build_monthly_estimates(data, window=504, gammas=[1.5, 3.0, 5.0])
    daily, _ = run_backtest(data, estimates)
    for frame, name in ((estimates, "estimates"), (daily, "daily")):
        values = frame.select_dtypes(include=[float, int]).to_numpy()
        if not np.isfinite(values).all():
            raise AssertionError(f"{name} contains NaN or Inf")


def test_empirical_deterministic_with_same_data():
    data = synthetic_empirical_data()
    estimates_1 = build_monthly_estimates(data, window=504, gammas=[1.5, 3.0, 5.0])
    daily_1, turnover_1 = run_backtest(data, estimates_1)
    metrics_1 = compute_metrics(daily_1, turnover_1)
    estimates_2 = build_monthly_estimates(data, window=504, gammas=[1.5, 3.0, 5.0])
    daily_2, turnover_2 = run_backtest(data, estimates_2)
    metrics_2 = compute_metrics(daily_2, turnover_2)
    pd.testing.assert_frame_equal(estimates_1.reset_index(drop=True), estimates_2.reset_index(drop=True))
    pd.testing.assert_frame_equal(daily_1.reset_index(drop=True), daily_2.reset_index(drop=True))
    pd.testing.assert_frame_equal(metrics_1.reset_index(drop=True), metrics_2.reset_index(drop=True))


def test_call_payoff_correct():
    s = np.array([80.0, 100.0, 120.0])
    np.testing.assert_allclose(call_payoff(s, k=100.0), np.array([0.0, 0.0, 20.0]))


def test_bs_delta_between_zero_and_one():
    s = np.linspace(50.0, 150.0, 101)
    delta = bs_delta(s, tau=0.5, k=100.0, r=0.02, sigma=0.20)
    if not ((delta >= 0.0) & (delta <= 1.0)).all():
        raise AssertionError("Black-Scholes delta left [0, 1]")


def test_transaction_cost_zero_if_delta_unchanged_positive_otherwise():
    same = transaction_cost(0.01, 100.0, 0.4, 0.4)
    changed = transaction_cost(0.01, 100.0, 0.6, 0.4)
    assert_close(float(same), 0.0, 1e-15, "transaction cost unchanged")
    if not float(changed) > 0.0:
        raise AssertionError("Transaction cost did not become positive after a trade")


def test_hedge_no_cost_error_decreases_with_refinement_on_average():
    times, paths = simulate_gbm_paths(n_paths=12000, n_steps=252, mu=0.08, seed=2024)
    rmses = []
    for frequency in (12, 52, 252):
        result = simulate_hedge(paths, times, lam=0.0, frequency=frequency, strategy="delta_bs")
        rmses.append(hedge_metrics_from_result(result, lam=0.0, frequency=frequency, strategy="delta_bs", band=0.0)["rmse"])
    if not (rmses[2] < rmses[1] < rmses[0]):
        raise AssertionError(f"No-cost hedge RMSE did not decrease with refinement: {rmses}")


def test_hedge_no_nan_inf_and_reproducible_seed():
    times_1, paths_1 = simulate_gbm_paths(n_paths=2000, n_steps=52, seed=777)
    times_2, paths_2 = simulate_gbm_paths(n_paths=2000, n_steps=52, seed=777)
    np.testing.assert_allclose(times_1, times_2)
    np.testing.assert_allclose(paths_1, paths_2)
    result_1 = simulate_hedge(paths_1, times_1, lam=0.002, frequency=52, strategy="no_trade_band", band=0.05)
    result_2 = simulate_hedge(paths_2, times_2, lam=0.002, frequency=52, strategy="no_trade_band", band=0.05)
    for key in ("error", "terminal_wealth", "payoff", "cumulative_cost", "turnover", "n_trades"):
        if not np.isfinite(result_1[key]).all():
            raise AssertionError(f"Hedge result {key} contains NaN or Inf")
        np.testing.assert_allclose(result_1[key], result_2[key])


def test_bellman_policy_stays_on_admissible_grid_and_is_finite():
    grids = bellman.make_grids(n_s=7, n_b=9, n_q=7)
    sol = bellman.solve_bellman(0.0005, n_steps=2, grids=grids)
    q_min, q_max = sol["q_grid"][0], sol["q_grid"][-1]
    for policy in sol["policies"]:
        if not np.isfinite(policy).all():
            raise AssertionError("Bellman policy contains NaN or Inf")
        if not ((policy >= q_min) & (policy <= q_max)).all():
            raise AssertionError("Bellman policy left the admissible action grid")
    for value in sol["values"]:
        if not np.isfinite(value).all():
            raise AssertionError("Bellman value contains NaN or Inf")


def test_bellman_reproducible():
    grids = bellman.make_grids(n_s=7, n_b=9, n_q=7)
    sol_1 = bellman.solve_bellman(0.002, n_steps=2, grids=grids)
    sol_2 = bellman.solve_bellman(0.002, n_steps=2, grids=grids)
    for p1, p2 in zip(sol_1["policies"], sol_2["policies"]):
        np.testing.assert_allclose(p1, p2)
    for v1, v2 in zip(sol_1["values"], sol_2["values"]):
        np.testing.assert_allclose(v1, v2)


def test_bellman_state_and_action_grids_are_independent():
    grids = bellman.make_grids(n_s=7, n_b=9, n_q=5)
    action_grid = np.linspace(0.0, 1.0, 13)
    sol = bellman.solve_bellman(0.002, n_steps=2, grids=grids, n_quad=3, action_grid=action_grid)
    np.testing.assert_allclose(sol["action_grid"], action_grid)
    allowed = np.isin(np.concatenate([p.ravel() for p in sol["policies"]]), action_grid)
    if not allowed.all():
        raise AssertionError("A policy action did not belong to the independent action grid")
    if len(sol["q_grid"]) == len(sol["action_grid"]):
        raise AssertionError("The test did not actually separate state and action grids")


def test_empirical_cash_grid_is_strict_and_has_safety_margin():
    samples = np.r_[np.full(500, 8.0), np.linspace(-100.0, 5.0, 1000)]
    grid = bellman.quantile_cash_grid(samples, n_b=31)
    if not np.all(np.diff(grid) > 0.0):
        raise AssertionError("Empirical cash grid is not strictly increasing")
    if not (grid[0] < samples.min() and grid[-1] > samples.max()):
        raise AssertionError("Empirical cash grid does not include a safety margin")


def test_bellman_lambda_zero_policy_reasonably_close_to_delta_on_coarse_grid():
    grids = bellman.make_grids(n_s=9, n_b=11, n_q=11)
    sol = bellman.solve_bellman(0.0, n_steps=3, grids=grids)
    metrics = bellman.policy_vs_delta_metrics(sol)
    if not metrics["rmse_policy_vs_delta"].mean() < 0.6:
        raise AssertionError("Coarse Bellman lambda=0 policy is unexpectedly far from BS delta")


def test_bellman_solver_does_not_call_bs_delta_during_optimization():
    original = bellman.bs_delta

    def forbidden(*args, **kwargs):
        raise AssertionError("bs_delta was called during Bellman optimization")

    bellman.bs_delta = forbidden
    try:
        grids = bellman.make_grids(n_s=5, n_b=7, n_q=5)
        bellman.solve_bellman(0.0, n_steps=1, grids=grids)
        cost_grids = bellman.make_cost_grids(n_s=5, n_b=7, n_q=5)
        bellman.solve_bellman(0.002, n_steps=1, grids=cost_grids)
    finally:
        bellman.bs_delta = original


def test_bellman_one_step_recurrence_matches_direct_backup():
    grids = bellman.make_grids(n_s=5, n_b=7, n_q=5)
    sol = bellman.solve_bellman(0.0005, n_steps=1, grids=grids)
    s_grid, b_grid, q_grid = sol["s_grid"], sol["b_grid"], sol["q_grid"]
    iq, is_, ib = 2, 2, 3
    direct = []
    for ia, action in enumerate(q_grid):
        direct.append(
            bellman.bellman_backup_state(
                s_grid[is_],
                b_grid[ib],
                q_grid[iq],
                action,
                0.0005,
                sol["values"][1][ia],
                s_grid,
                b_grid,
                sol["dt"],
            )
        )
    np.testing.assert_allclose(sol["values"][0][iq, is_, ib], min(direct), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sol["policies"][0][iq, is_, ib], q_grid[int(np.argmin(direct))])


def test_bellman_interpolation_consistent_on_grid_node():
    s_grid = np.array([90.0, 100.0, 110.0])
    b_grid = np.array([-10.0, 0.0, 10.0])
    values = np.arange(9, dtype=float).reshape(3, 3)
    got_linear = bellman.interp2_on_grid(values, s_grid, b_grid, 100.0, 0.0, method="linear")
    got_quad = bellman.interp2_on_grid(values, s_grid, b_grid, 100.0, 0.0, method="quadratic_cash")
    assert_close(float(got_linear), values[1, 1], 1e-12, "linear interpolation at node")
    assert_close(float(got_quad), values[1, 1], 1e-12, "quadratic cash interpolation at node")


def test_bellman_terminal_value_convex_in_cash():
    grids = bellman.make_cost_grids(n_s=5, n_b=9, n_q=5)
    sol = bellman.solve_bellman(0.002, n_steps=1, grids=grids)
    terminal = sol["values"][-1]
    second_diff = np.diff(terminal, n=2, axis=2)
    if not (second_diff >= -1e-10).all():
        raise AssertionError("Terminal Bellman value is not convex in cash")


def test_transaction_cost_increases_with_trade_size():
    small = transaction_cost(0.002, 100.0, 0.55, 0.50)
    large = transaction_cost(0.002, 100.0, 0.80, 0.50)
    if not float(large) > float(small) > 0.0:
        raise AssertionError("Transaction cost did not increase with |a-q|")


def test_bellman_refined_mini_grid_stable_and_finite():
    row, _, _ = bellman.cost_solution_metrics(n_s=11, n_b=13, n_q=9, n_steps=2, n_paths=100)
    numeric = np.array([v for v in row.values() if isinstance(v, (int, float, np.floating))], dtype=float)
    if not np.isfinite(numeric).all():
        raise AssertionError("Cost Bellman mini-grid diagnostics contain NaN or Inf")
    if row["Nq"] != 9 or row["Ns"] != 11:
        raise AssertionError("Cost Bellman mini-grid returned inconsistent metadata")
    for key in ["fraction_cash_outside_grid", "fraction_s_outside_grid"]:
        if not 0.0 <= row[key] <= 1.0:
            raise AssertionError(f"Invalid extrapolation diagnostic {key}")


def test_barrier_down_out_payoff_and_knockout_zero():
    s = np.array([90.0, 110.0, 130.0])
    alive = np.array([False, True, False])
    np.testing.assert_allclose(barrier.down_out_call_payoff(s, alive), np.array([0.0, 10.0, 0.0]))


def test_barrier_knockout_is_irreversible_in_simulation():
    times, paths, alive_bridge, _ = barrier.simulate_barrier_paths(n_paths=500, n_steps=52, seed=99)
    del times, paths
    revived = np.any((~alive_bridge[:, :-1]) & alive_bridge[:, 1:])
    if revived:
        raise AssertionError("Barrier state reactivated after knock-out")


def test_barrier_bridge_probability_simple_cases():
    p_hit_endpoint = barrier.bridge_hit_probability(100.0, 70.0, 1.0 / 12.0)
    assert_close(float(p_hit_endpoint), 1.0, 1e-14, "endpoint below barrier must hit")
    p_far = float(barrier.bridge_hit_probability(140.0, 140.0, 1.0 / 252.0))
    p_near = float(barrier.bridge_hit_probability(82.0, 82.0, 1.0 / 252.0))
    if not 0.0 <= p_far < p_near < 1.0:
        raise AssertionError("Brownian bridge crossing probability is not ordered near/far from barrier")


def test_barrier_policy_grid_finite_and_admissible():
    grids = barrier.make_cost_grids(n_s=5, n_b=7, n_q=5)
    sol = barrier.solve_bellman_costs(0.002, n_steps=1, grids=grids, n_quad=3)
    q_min, q_max = sol["q_grid"][0], sol["q_grid"][-1]
    for policy in sol["policies"]:
        if not np.isfinite(policy).all():
            raise AssertionError("Barrier Bellman policy contains NaN or Inf")
        if not ((policy >= q_min) & (policy <= q_max)).all():
            raise AssertionError("Barrier Bellman policy leaves admissible q grid")


def test_barrier_solver_does_not_use_benchmark_during_optimization():
    original_delta = barrier.barrier_delta_numeric
    original_price = barrier.barrier_price

    def forbidden(*args, **kwargs):
        raise AssertionError("Barrier benchmark was called during Bellman optimization")

    barrier.barrier_delta_numeric = forbidden
    barrier.barrier_price = forbidden
    try:
        barrier.solve_lambda0_reduced(n_steps=1, n_s=7, n_quad=3)
        grids = barrier.make_cost_grids(n_s=5, n_b=7, n_q=5)
        barrier.solve_bellman_costs(0.002, n_steps=1, grids=grids, n_quad=3)
    finally:
        barrier.barrier_delta_numeric = original_delta
        barrier.barrier_price = original_price


def test_barrier_one_step_recurrence_matches_direct_backup():
    grids = barrier.make_cost_grids(n_s=5, n_b=7, n_q=5)
    sol = barrier.solve_bellman_costs(0.0005, n_steps=1, grids=grids, n_quad=3)
    s_grid, b_grid, q_grid = sol["s_grid"], sol["b_grid"], sol["q_grid"]
    ialive, iq, is_, ib = 1, 2, 2, 3
    direct = []
    for ia, action in enumerate(q_grid):
        direct.append(
            barrier.bellman_backup_cost(
                s_grid[is_],
                b_grid[ib],
                q_grid[iq],
                action,
                True,
                0.0005,
                sol["values"][1][:, ia],
                s_grid,
                b_grid,
                sol["dt"],
                n_quad=3,
            )
        )
    np.testing.assert_allclose(sol["values"][0][ialive, iq, is_, ib], min(direct), rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    test_merton_identity()
    test_terminal_condition_is_crra_utility()
    test_analytic_value_formula_matches_requested_benchmark()
    test_gamma_3_analytic_control_is_half_x()
    test_log_transform_vxx()
    test_interpolation_at_y0_matches_odd_node()
    test_interpolation_at_x0_between_nodes_is_linear()
    test_hjb_solution_has_no_nan_or_inf()
    test_empirical_no_future_information_used()
    test_empirical_first_backtest_date_after_504_observations()
    test_empirical_long_only_weights_are_bounded()
    test_empirical_cash_benchmark_follows_risk_free_rate()
    test_empirical_no_nan_inf_in_traded_periods()
    test_empirical_deterministic_with_same_data()
    test_call_payoff_correct()
    test_bs_delta_between_zero_and_one()
    test_transaction_cost_zero_if_delta_unchanged_positive_otherwise()
    test_hedge_no_cost_error_decreases_with_refinement_on_average()
    test_hedge_no_nan_inf_and_reproducible_seed()
    test_bellman_policy_stays_on_admissible_grid_and_is_finite()
    test_bellman_reproducible()
    test_bellman_lambda_zero_policy_reasonably_close_to_delta_on_coarse_grid()
    test_bellman_solver_does_not_call_bs_delta_during_optimization()
    test_bellman_one_step_recurrence_matches_direct_backup()
    test_bellman_interpolation_consistent_on_grid_node()
    test_bellman_terminal_value_convex_in_cash()
    test_transaction_cost_increases_with_trade_size()
    test_bellman_refined_mini_grid_stable_and_finite()
    test_barrier_down_out_payoff_and_knockout_zero()
    test_barrier_knockout_is_irreversible_in_simulation()
    test_barrier_bridge_probability_simple_cases()
    test_barrier_policy_grid_finite_and_admissible()
    test_barrier_solver_does_not_use_benchmark_during_optimization()
    test_barrier_one_step_recurrence_matches_direct_backup()
    print("All core tests passed.")
