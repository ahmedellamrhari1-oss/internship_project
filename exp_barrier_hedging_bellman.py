"""
Experience 4: Bellman hedging for a Down-and-Out Call barrier option.

Discrete dynamics at decision dates t_n=n dt:

    S_{n+1} = S_n exp((mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z)

The barrier state is I_n=1 while the option is alive and I_n=0 after knock-out.
Knock-out is irreversible. If both endpoints are above B, the conditional
Brownian-bridge survival probability over one interval is

    p_surv = 1 - exp(-2 log(S_n/B) log(S_{n+1}/B) / (sigma^2 dt)).

If either endpoint is below the barrier, p_surv=0. This correction is used in
the Bellman transition and in Monte Carlo backtests; a naive endpoint diagnostic
is saved separately.

For lambda=0 the value is represented exactly as a quadratic in total wealth:

    V_n(S,W,I) = A_n(S,I) W^2 - 2 B_n(S,I) W + C_n(S,I).

For a fixed state, W_{n+1}=exp(r dt) W_n + a(S_{n+1}-exp(r dt)S_n), so the
optimal action is obtained analytically from the quadratic expectation. This is
a discrete stochastic-control reduction, not a continuous HJB/QVI solver.

For lambda>0, transaction costs

    C_n = lambda S_n |a_n-q_n|
    b_n^+ = b_n - (a_n-q_n)S_n - C_n
    b_{n+1} = b_n^+ exp(r dt)

break the global quadratic wealth reduction. The implemented solver therefore
uses a moderate explicit grid in (I,q,S,b), with a non-uniform cash grid. The
control is the next stock position a=q'. No benchmark delta or analytic barrier
price is used inside the Bellman optimizer.
"""

import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from exp_hedging_transaction_costs import norm_cdf, bs_call_price, bs_delta, transaction_cost


RESULT_DIR = Path("results") / "barrier_hedging_bellman"
FIG_DIR = RESULT_DIR / "figures"

S0 = 100.0
K = 100.0
BARRIER = 80.0
T = 1.0
R = 0.02
MU = 0.08
SIGMA = 0.20
SEED = 24680
N_FINE = 252
N_PATHS = 2000
LAMBDAS = [0.0, 0.0005, 0.002, 0.01]
GH_CACHE = {}


def ensure_dirs():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def gauss_hermite_normal(n_quad):
    if n_quad not in GH_CACHE:
        nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
        GH_CACHE[n_quad] = math.sqrt(2.0) * nodes, weights / math.sqrt(math.pi)
    return GH_CACHE[n_quad]


def down_out_call_payoff(s, alive, k=K):
    return np.maximum(np.asarray(s, dtype=float) - k, 0.0) * np.asarray(alive, dtype=float)


def bridge_survival_probability(s0, s1, dt, barrier=BARRIER, sigma=SIGMA):
    s0 = np.asarray(s0, dtype=float)
    s1 = np.asarray(s1, dtype=float)
    out = np.zeros(np.broadcast_shapes(s0.shape, s1.shape), dtype=float)
    s0b, s1b = np.broadcast_arrays(s0, s1)
    mask = (s0b > barrier) & (s1b > barrier) & (dt > 0.0)
    exponent = -2.0 * np.log(s0b[mask] / barrier) * np.log(s1b[mask] / barrier) / (sigma ** 2 * dt)
    out[mask] = 1.0 - np.exp(np.minimum(exponent, 0.0))
    return np.clip(out, 0.0, 1.0)


def bridge_hit_probability(s0, s1, dt, barrier=BARRIER, sigma=SIGMA):
    return 1.0 - bridge_survival_probability(s0, s1, dt, barrier, sigma)


def make_s_grid(n_s=151):
    return np.exp(np.linspace(np.log(BARRIER * 0.72), np.log(260.0), n_s))


def make_cost_grids(n_s=35, n_b=41, n_q=17):
    s_grid = make_s_grid(n_s)
    q_grid = np.linspace(-0.75, 1.25, n_q)
    u = np.linspace(-1.0, 1.0, n_b)
    center = 5.0
    half_width = 250.0
    b_grid = center + half_width * np.sinh(1.7 * u) / np.sinh(1.7)
    b_grid[0], b_grid[-1] = -280.0, 260.0
    return s_grid, np.unique(np.sort(b_grid)), q_grid


def interp1(values, grid, x):
    return np.interp(np.asarray(x, dtype=float), grid, values, left=values[0], right=values[-1])


def interp2(values, s_grid, b_grid, s, b):
    s_arr, b_arr = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(b, dtype=float))
    sf = s_arr.reshape(-1)
    bf = b_arr.reshape(-1)
    out = np.empty_like(sf)
    for k, (sv, bv) in enumerate(zip(sf, bf)):
        row = np.array([np.interp(bv, b_grid, values[i], left=values[i, 0], right=values[i, -1]) for i in range(len(s_grid))])
        out[k] = np.interp(sv, s_grid, row, left=row[0], right=row[-1])
    return out.reshape(s_arr.shape)


def barrier_price_grid(n_steps=80, n_s=401, n_quad=11, drift=R, discount=True):
    """Risk-neutral numerical price benchmark with Brownian-bridge survival."""
    s_grid = make_s_grid(n_s)
    dt = T / n_steps
    z, w = gauss_hermite_normal(n_quad)
    values = np.where(s_grid > BARRIER, np.maximum(s_grid - K, 0.0), 0.0)
    profiles = [(T, values.copy())]
    disc = math.exp(-R * dt) if discount else 1.0
    for n in range(n_steps - 1, -1, -1):
        now = np.zeros_like(s_grid)
        for i, s in enumerate(s_grid):
            if s <= BARRIER:
                now[i] = 0.0
                continue
            sn = s * np.exp((drift - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z)
            surv = bridge_survival_probability(s, sn, dt)
            cont = interp1(values, s_grid, sn)
            now[i] = disc * float(np.sum(w * surv * cont))
        values = now
        if n in {0, n_steps // 2, n_steps - 1}:
            profiles.append((n * dt, values.copy()))
    return {"s_grid": s_grid, "values": values, "profiles": profiles[::-1], "n_steps": n_steps, "n_quad": n_quad}


def barrier_benchmark_tree(n_steps=24, n_s=301, n_quad=7):
    """Risk-neutral value and finite-difference delta at every decision date."""
    s_grid = make_s_grid(n_s)
    dt = T / n_steps
    z, w = gauss_hermite_normal(n_quad)
    values = [None] * (n_steps + 1)
    values[n_steps] = np.where(s_grid > BARRIER, np.maximum(s_grid - K, 0.0), 0.0)
    disc = math.exp(-R * dt)
    for n in range(n_steps - 1, -1, -1):
        now = np.zeros_like(s_grid)
        for i, s in enumerate(s_grid):
            if s <= BARRIER:
                continue
            sn = s * np.exp((R - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z)
            surv = bridge_survival_probability(s, sn, dt)
            now[i] = disc * float(np.sum(w * surv * interp1(values[n + 1], s_grid, sn)))
        values[n] = now
    deltas = []
    for val in values:
        delta = np.gradient(val, s_grid)
        deltas.append(np.clip(delta, -2.0, 2.0))
    return {"s_grid": s_grid, "values": values, "deltas": deltas, "n_steps": n_steps}


def barrier_price(s, tau, benchmark=None):
    s = np.asarray(s, dtype=float)
    if np.all(np.asarray(tau) <= 1e-14):
        return np.where(s > BARRIER, np.maximum(s - K, 0.0), 0.0)
    tau_scalar = float(np.asarray(tau).reshape(-1)[0])
    steps = max(2, int(round(80 * tau_scalar / T)))
    bench = barrier_price_grid(n_steps=steps, n_s=401, n_quad=9, drift=R, discount=True) if benchmark is None else benchmark
    return interp1(bench["values"], bench["s_grid"], s)


def barrier_delta_numeric(s, tau):
    s = np.asarray(s, dtype=float)
    tau = float(tau)
    h = np.maximum(0.25, 0.005 * s)
    up = np.maximum(s + h, BARRIER + 1e-6)
    dn = np.maximum(s - h, BARRIER + 1e-6)
    pu = barrier_price(up, tau)
    pdn = barrier_price(dn, tau)
    denom = np.maximum(up - dn, 1e-12)
    return np.clip((pu - pdn) / denom, -2.0, 2.0)


def solve_lambda0_reduced(n_steps=24, n_s=201, n_quad=9, drift=MU):
    s_grid = make_s_grid(n_s)
    dt = T / n_steps
    rf = math.exp(R * dt)
    z, w = gauss_hermite_normal(n_quad)
    payoff_alive = np.where(s_grid > BARRIER, np.maximum(s_grid - K, 0.0), 0.0)
    coeff_alive = (np.ones_like(s_grid), payoff_alive.copy(), payoff_alive ** 2)
    coeff_dead = (np.ones_like(s_grid), np.zeros_like(s_grid), np.zeros_like(s_grid))
    coeffs = [(coeff_alive, coeff_dead)]
    policies_w0 = []

    for _ in range(n_steps - 1, -1, -1):
        next_alive, next_dead = coeff_alive, coeff_dead
        alive_now = [np.empty_like(s_grid) for _ in range(3)]
        dead_now = [np.empty_like(s_grid) for _ in range(3)]
        pol_alive = np.zeros_like(s_grid)
        pol_dead = np.zeros_like(s_grid)
        for alive_state, target, pol in [(1, alive_now, pol_alive), (0, dead_now, pol_dead)]:
            for i, s in enumerate(s_grid):
                sn = s * np.exp((drift - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z)
                if alive_state and s > BARRIER:
                    ps = bridge_survival_probability(s, sn, dt)
                else:
                    ps = np.zeros_like(sn)
                A = ps * interp1(next_alive[0], s_grid, sn) + (1.0 - ps) * interp1(next_dead[0], s_grid, sn)
                Bb = ps * interp1(next_alive[1], s_grid, sn) + (1.0 - ps) * interp1(next_dead[1], s_grid, sn)
                Cc = ps * interp1(next_alive[2], s_grid, sn) + (1.0 - ps) * interp1(next_dead[2], s_grid, sn)
                y = sn - rf * s
                e_a = float(np.sum(w * A))
                e_b = float(np.sum(w * Bb))
                e_c = float(np.sum(w * Cc))
                e_ay = float(np.sum(w * A * y))
                e_by = float(np.sum(w * Bb * y))
                e_ayy = max(float(np.sum(w * A * y ** 2)), 1e-12)
                target[0][i] = rf ** 2 * e_a - (rf * e_ay) ** 2 / e_ayy
                target[1][i] = rf * e_b - (rf * e_ay * e_by) / e_ayy
                target[2][i] = e_c - (e_by ** 2) / e_ayy
                pol[i] = np.clip(e_by / e_ayy, -0.75, 1.25)
        coeff_alive = tuple(alive_now)
        coeff_dead = tuple(dead_now)
        coeffs.append((coeff_alive, coeff_dead))
        policies_w0.append((pol_alive, pol_dead))
    return {
        "kind": "lambda0_reduced",
        "n_steps": n_steps,
        "dt": dt,
        "s_grid": s_grid,
        "coeffs": list(reversed(coeffs)),
        "policies_w0": list(reversed(policies_w0)),
        "n_quad": n_quad,
        "drift": drift,
    }


def lambda0_action(solution, n, s, wealth, alive):
    s_grid = solution["s_grid"]
    dt = solution["dt"]
    rf = math.exp(R * dt)
    z, w = gauss_hermite_normal(solution["n_quad"])
    next_alive, next_dead = solution["coeffs"][n + 1]
    s_arr, w_arr, alive_arr = np.broadcast_arrays(np.asarray(s, dtype=float), np.asarray(wealth, dtype=float), np.asarray(alive, dtype=bool))
    out = np.zeros(s_arr.size, dtype=float)
    for j, (sv, wv, av) in enumerate(zip(s_arr.reshape(-1), w_arr.reshape(-1), alive_arr.reshape(-1))):
        sn = sv * np.exp((solution["drift"] - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z)
        ps = bridge_survival_probability(sv, sn, dt) if av and sv > BARRIER else np.zeros_like(sn)
        A = ps * interp1(next_alive[0], s_grid, sn) + (1.0 - ps) * interp1(next_dead[0], s_grid, sn)
        Bb = ps * interp1(next_alive[1], s_grid, sn) + (1.0 - ps) * interp1(next_dead[1], s_grid, sn)
        y = sn - rf * sv
        e_ay = float(np.sum(w * A * y))
        e_by = float(np.sum(w * Bb * y))
        e_ayy = max(float(np.sum(w * A * y ** 2)), 1e-12)
        out[j] = np.clip((e_by - wv * rf * e_ay) / e_ayy, -0.75, 1.25)
    return out.reshape(s_arr.shape)


def bellman_backup_cost(s, b, q, action, alive, lam, next_values, s_grid, b_grid, dt, n_quad=5, drift=MU):
    rf = math.exp(R * dt)
    z, w = gauss_hermite_normal(n_quad)
    sn = s * np.exp((drift - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z)
    cost = transaction_cost(lam, s, action, q)
    bn = (b - (action - q) * s - cost) * rf
    v_dead = interp2(next_values[0], s_grid, b_grid, sn, bn)
    if alive and s > BARRIER:
        ps = bridge_survival_probability(s, sn, dt)
        v_alive = interp2(next_values[1], s_grid, b_grid, sn, bn)
        val = ps * v_alive + (1.0 - ps) * v_dead
    else:
        val = v_dead
    return float(np.sum(w * val))


def solve_bellman_costs(lam, n_steps=8, grids=None, n_quad=5, drift=MU):
    if grids is None:
        grids = make_cost_grids()
    s_grid, b_grid, q_grid = grids
    dt = T / n_steps
    terminal = np.zeros((2, len(q_grid), len(s_grid), len(b_grid)), dtype=float)
    for iq, q in enumerate(q_grid):
        wealth = b_grid[None, :] + q * s_grid[:, None]
        payoff_alive = np.maximum(s_grid[:, None] - K, 0.0) * (s_grid[:, None] > BARRIER)
        terminal[1, iq] = (wealth - payoff_alive) ** 2
        terminal[0, iq] = wealth ** 2
    values = [terminal]
    policies = []
    v_next = terminal
    for _ in range(n_steps - 1, -1, -1):
        v_now = np.empty_like(terminal)
        p_now = np.empty_like(terminal)
        for ialive, alive in enumerate([False, True]):
            for iq, q in enumerate(q_grid):
                for is_, s in enumerate(s_grid):
                    for ib, b in enumerate(b_grid):
                        candidates = [
                            bellman_backup_cost(s, b, q, a, alive, lam, v_next[:, ia], s_grid, b_grid, dt, n_quad=n_quad, drift=drift)
                            for ia, a in enumerate(q_grid)
                        ]
                        best = int(np.argmin(candidates))
                        v_now[ialive, iq, is_, ib] = candidates[best]
                        p_now[ialive, iq, is_, ib] = q_grid[best]
        values.append(v_now)
        policies.append(p_now)
        v_next = v_now
    return {
        "kind": "cost_grid",
        "lambda": lam,
        "n_steps": n_steps,
        "dt": dt,
        "s_grid": s_grid,
        "b_grid": b_grid,
        "q_grid": q_grid,
        "values": list(reversed(values)),
        "policies": list(reversed(policies)),
        "n_quad": n_quad,
        "drift": drift,
    }


def policy_action_cost(solution, n, s, b, q, alive):
    if not alive:
        ialive = 0
    else:
        ialive = 1
    q_grid = solution["q_grid"]
    iq = int(np.abs(q_grid - q).argmin())
    return float(interp2(solution["policies"][n][ialive, iq], solution["s_grid"], solution["b_grid"], s, b))


def simulate_barrier_paths(n_paths=N_PATHS, n_steps=N_FINE, seed=SEED, mu=MU, bridge=True):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_inc = (mu - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z
    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_inc, axis=1))
    alive_naive = np.minimum.accumulate(paths, axis=1) > BARRIER
    alive_bridge = np.ones_like(paths, dtype=bool)
    if bridge:
        u = rng.random((n_paths, n_steps))
        alive = np.ones(n_paths, dtype=bool)
        for i in range(n_steps):
            ps = bridge_survival_probability(paths[:, i], paths[:, i + 1], dt)
            hit = (~(paths[:, i + 1] > BARRIER)) | (u[:, i] > ps)
            alive &= ~hit
            alive_bridge[:, i + 1] = alive
    else:
        alive_bridge = alive_naive
    return np.linspace(0.0, T, n_steps + 1), paths, alive_bridge, alive_naive


def rebalance_indices(frequency, n_steps=N_FINE):
    return np.unique(np.rint(np.linspace(0, n_steps, frequency + 1)).astype(int))


def simulate_strategy(paths, times, alive_path, lam, frequency, strategy, solution=None, band=0.0, store_paths=False, initial_price=None, barrier_tree=None):
    n_paths = paths.shape[0]
    idx = rebalance_indices(frequency, paths.shape[1] - 1)
    if initial_price is None:
        tree = barrier_benchmark_tree(n_steps=frequency)
        initial_price = float(interp1(tree["values"][0], tree["s_grid"], S0))
    if barrier_tree is None and strategy in {"barrier_delta", "barrier_no_trade"}:
        barrier_tree = barrier_benchmark_tree(n_steps=frequency)
    cash = np.full(n_paths, initial_price, dtype=float)
    q = np.zeros(n_paths, dtype=float)
    costs = np.zeros(n_paths, dtype=float)
    turnover = np.zeros(n_paths, dtype=float)
    trades = np.zeros(n_paths, dtype=float)
    q_hist = np.full((n_paths, len(idx) - 1), np.nan) if store_paths else None
    delta_hist = np.full_like(q_hist, np.nan) if store_paths else None
    last_i = 0
    for j, i in enumerate(idx[:-1]):
        if i > last_i:
            cash *= np.exp(R * (times[i] - times[last_i]))
        s = paths[:, i]
        alive = alive_path[:, i]
        tau = T - times[i]
        if strategy == "bellman_lambda0":
            wealth = cash + q * s
            new_q = lambda0_action(solution, min(j, solution["n_steps"] - 1), s, wealth, alive)
        elif strategy == "bellman_costs":
            new_q = np.array([policy_action_cost(solution, min(j, solution["n_steps"] - 1), s[k], cash[k], q[k], bool(alive[k])) for k in range(n_paths)])
        elif strategy == "european_delta":
            new_q = np.where(alive, bs_delta(s, tau), 0.0)
        elif strategy == "barrier_delta":
            delta_b = interp1(barrier_tree["deltas"][min(j, barrier_tree["n_steps"])], barrier_tree["s_grid"], s)
            new_q = np.where(alive, delta_b, 0.0)
        elif strategy == "barrier_no_trade":
            delta_b = interp1(barrier_tree["deltas"][min(j, barrier_tree["n_steps"])], barrier_tree["s_grid"], s)
            target = np.where(alive, delta_b, 0.0)
            new_q = np.where(np.abs(target - q) > band, target, q)
        else:
            raise ValueError(strategy)
        dc = transaction_cost(lam, s, new_q, q)
        cash -= (new_q - q) * s + dc
        costs += dc
        turnover += np.abs(new_q - q)
        trades += np.abs(new_q - q) > 1e-12
        q = new_q
        if store_paths:
            q_hist[:, j] = q
            if barrier_tree is not None:
                delta_hist[:, j] = np.where(alive, interp1(barrier_tree["deltas"][min(j, barrier_tree["n_steps"])], barrier_tree["s_grid"], s), 0.0)
            else:
                delta_hist[:, j] = np.where(alive, barrier_delta_numeric(s, tau), 0.0)
        last_i = i
    cash *= np.exp(R * (T - times[last_i]))
    wealth = cash + q * paths[:, -1]
    payoff = down_out_call_payoff(paths[:, -1], alive_path[:, -1])
    error = wealth - payoff
    near = np.min(paths, axis=1) <= BARRIER * 1.10
    return {
        "error": error,
        "pnl": error,
        "terminal_wealth": wealth,
        "payoff": payoff,
        "cumulative_cost": costs,
        "turnover": turnover,
        "n_trades": trades,
        "alive_terminal": alive_path[:, -1],
        "near_barrier": near,
        "rebalance_indices": idx,
        "held_delta_path": q_hist,
        "barrier_delta_path": delta_hist,
    }


def metrics_from_result(result, lam, frequency, strategy, band=-1.0):
    e = result["error"]
    rows = {
        "lambda": lam,
        "frequency": frequency,
        "strategy": strategy,
        "band": band,
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "q01": float(np.quantile(e, 0.01)),
        "q05": float(np.quantile(e, 0.05)),
        "q50": float(np.quantile(e, 0.50)),
        "q95": float(np.quantile(e, 0.95)),
        "q99": float(np.quantile(e, 0.99)),
        "mean_cost": float(np.mean(result["cumulative_cost"])),
        "turnover": float(np.mean(result["turnover"])),
        "n_trades": float(np.mean(result["n_trades"])),
        "knockout_fraction": float(np.mean(~result["alive_terminal"])),
    }
    for name, mask in [("knockout", ~result["alive_terminal"]), ("survivor", result["alive_terminal"]), ("near_barrier", result["near_barrier"])]:
        rows[f"rmse_{name}"] = float(np.sqrt(np.mean(e[mask] ** 2))) if np.any(mask) else np.nan
    return rows


def classify_buy_hold_sell(solution):
    rows = []
    if solution["kind"] != "cost_grid":
        return pd.DataFrame(rows)
    q_grid = solution["q_grid"]
    eps = 0.5 * np.min(np.diff(q_grid))
    for n, pol in enumerate(solution["policies"]):
        for ialive, alive in enumerate([0, 1]):
            for iq, q in enumerate(q_grid):
                diff = pol[ialive, iq] - q
                rows.append({
                    "lambda": solution["lambda"],
                    "t": n * solution["dt"],
                    "alive": alive,
                    "q": q,
                    "buy_fraction": float(np.mean(diff > eps)),
                    "hold_fraction": float(np.mean(np.abs(diff) <= eps)),
                    "sell_fraction": float(np.mean(diff < -eps)),
                    "near_barrier_hold": float(np.mean(np.abs(diff[solution["s_grid"] < BARRIER * 1.12]) <= eps)),
                    "far_hold": float(np.mean(np.abs(diff[solution["s_grid"] > 120.0]) <= eps)),
                })
    return pd.DataFrame(rows)


def policy_profiles(lambda0_solution, cost_solutions, trees):
    rows = []
    s_points = np.array([BARRIER * 1.01, BARRIER * 1.05, 100.0, 120.0, 150.0])
    for n in [0, lambda0_solution["n_steps"] // 2, lambda0_solution["n_steps"] - 1]:
        t = n * lambda0_solution["dt"]
        tau = T - t
        tree = trees[lambda0_solution["n_steps"]]
        for s in s_points:
            price = float(interp1(tree["values"][n], tree["s_grid"], s))
            rows.append({
                "lambda": 0.0,
                "t": t,
                "S": s,
                "alive": 1,
                "barrier_price": price,
                "barrier_delta": float(interp1(tree["deltas"][n], tree["s_grid"], s)),
                "european_delta": float(bs_delta(s, tau)),
                "bellman_position": float(lambda0_action(lambda0_solution, min(n, lambda0_solution["n_steps"] - 1), s, price, True)),
                "knockout_probability_next_step": float(bridge_hit_probability(s, s, lambda0_solution["dt"])),
            })
    for sol in cost_solutions:
        for n in [0, sol["n_steps"] // 2, sol["n_steps"] - 1]:
            t = n * sol["dt"]
            tau = T - t
            tree = trees[sol["n_steps"]]
            for s in s_points:
                price = float(interp1(tree["values"][n], tree["s_grid"], s))
                rows.append({
                    "lambda": sol["lambda"],
                    "t": t,
                    "S": s,
                    "alive": 1,
                    "barrier_price": price,
                    "barrier_delta": float(interp1(tree["deltas"][n], tree["s_grid"], s)),
                    "european_delta": float(bs_delta(s, tau)),
                    "bellman_position": policy_action_cost(sol, min(n, sol["n_steps"] - 1), s, price, 0.0, True),
                    "knockout_probability_next_step": float(bridge_hit_probability(s, s, sol["dt"])),
                })
    return pd.DataFrame(rows)


def sample_errors(result, lam, strategy, n=1000):
    m = min(n, len(result["error"]))
    return pd.DataFrame({
        "lambda": lam,
        "strategy": strategy,
        "path": np.arange(m),
        "error": result["error"][:m],
        "terminal_wealth": result["terminal_wealth"][:m],
        "payoff": result["payoff"][:m],
        "knocked_out": ~result["alive_terminal"][:m],
        "near_barrier": result["near_barrier"][:m],
    })


def run_backtests(lambda0_solution, cost_solutions, initial_price):
    times, paths, alive_bridge, alive_naive = simulate_barrier_paths()
    frequencies = sorted({lambda0_solution["n_steps"], *[sol["n_steps"] for sol in cost_solutions]})
    trees = {freq: barrier_benchmark_tree(n_steps=freq, n_s=251, n_quad=7) for freq in frequencies}
    rows = []
    samples = []
    trajectories = []
    for lam in LAMBDAS:
        sol = lambda0_solution if lam == 0.0 else next(s for s in cost_solutions if abs(s["lambda"] - lam) < 1e-15)
        strategies = []
        if lam == 0.0:
            strategies.append(("bellman_lambda0", -1.0, sol))
        else:
            strategies.append(("bellman_costs", -1.0, sol))
            strategies.append(("barrier_no_trade", 0.05, None))
        strategies.extend([("european_delta", 0.0, None), ("barrier_delta", 0.0, None)])
        for strategy, band, sol_obj in strategies:
            result = simulate_strategy(paths, times, alive_bridge, lam, sol["n_steps"], strategy, solution=sol_obj, band=band, store_paths=True, initial_price=initial_price, barrier_tree=trees[sol["n_steps"]])
            rows.append(metrics_from_result(result, lam, sol["n_steps"], strategy, band))
            samples.append(sample_errors(result, lam, strategy))
            if strategy.startswith("bellman"):
                idx = result["rebalance_indices"][:-1]
                for path_id in [0, int(np.argmax(~alive_bridge[:, -1]))]:
                    trajectories.append(pd.DataFrame({
                        "lambda": lam,
                        "path": path_id,
                        "t": times[idx],
                        "S": paths[path_id, idx],
                        "alive": alive_bridge[path_id, idx].astype(int),
                        "q_bellman": result["held_delta_path"][path_id],
                        "barrier_delta": result["barrier_delta_path"][path_id],
                    }))
    crossing = pd.DataFrame([{
        "method": "naive_endpoint",
        "knockout_fraction": float(np.mean(~alive_naive[:, -1])),
    }, {
        "method": "brownian_bridge",
        "knockout_fraction": float(np.mean(~alive_bridge[:, -1])),
    }, {
        "method": "bridge_minus_naive",
        "knockout_fraction": float(np.mean(~alive_bridge[:, -1]) - np.mean(~alive_naive[:, -1])),
    }])
    return pd.DataFrame(rows), pd.concat(samples, ignore_index=True), crossing, pd.concat(trajectories, ignore_index=True)


def run_convergence(initial_price):
    rows_t = []
    rows_s = []
    times, paths, alive_bridge, _ = simulate_barrier_paths(n_paths=2500, seed=SEED + 1)
    for n_steps in [6, 12, 24]:
        sol = solve_lambda0_reduced(n_steps=n_steps, n_s=151, n_quad=7)
        res = simulate_strategy(paths, times, alive_bridge, 0.0, n_steps, "bellman_lambda0", solution=sol, initial_price=initial_price)
        tree = barrier_benchmark_tree(n_steps=n_steps, n_s=251, n_quad=7)
        bench = simulate_strategy(paths, times, alive_bridge, 0.0, n_steps, "barrier_delta", initial_price=initial_price, barrier_tree=tree)
        row = metrics_from_result(res, 0.0, n_steps, "bellman_lambda0")
        row["rmse_barrier_delta"] = metrics_from_result(bench, 0.0, n_steps, "barrier_delta")["rmse"]
        rows_t.append(row)
    for n_s in [81, 151, 251]:
        sol = solve_lambda0_reduced(n_steps=12, n_s=n_s, n_quad=7)
        res = simulate_strategy(paths, times, alive_bridge, 0.0, 12, "bellman_lambda0", solution=sol, initial_price=initial_price)
        row = metrics_from_result(res, 0.0, 12, "bellman_lambda0")
        row["Ns"] = n_s
        rows_s.append(row)
    return pd.DataFrame(rows_t), pd.DataFrame(rows_s)


def plot_outputs(metrics, profiles, errors, crossing, holds, trajectories):
    for path in FIG_DIR.glob("*.png"):
        path.unlink()
    plt.figure(figsize=(7, 5))
    crossing.plot(kind="bar", x="method", y="knockout_fraction", legend=False, ax=plt.gca())
    plt.ylabel("knock-out fraction")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "barrier_crossing_bridge_vs_naive.png", dpi=150)
    plt.close()

    for t, df in profiles[profiles["lambda"].eq(0.0)].groupby("t"):
        plt.figure(figsize=(7, 5))
        plt.plot(df["S"], df["barrier_price"], marker="o", label="value")
        plt.axvline(BARRIER, color="black", linestyle="--", linewidth=1)
        plt.xlabel("S")
        plt.ylabel("price")
        plt.title(f"Barrier value t={t:.2f}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"value_profile_t_{t:.2f}.png", dpi=150)
        plt.close()

    plt.figure(figsize=(8, 5))
    df0 = profiles[(profiles["lambda"].eq(0.0)) & (profiles["t"].eq(0.0))]
    plt.plot(df0["S"], df0["bellman_position"], marker="o", label="Bellman")
    plt.plot(df0["S"], df0["barrier_delta"], marker="s", label="barrier delta")
    plt.plot(df0["S"], df0["european_delta"], marker="^", label="european delta")
    plt.axvline(BARRIER, color="black", linestyle="--", linewidth=1)
    plt.xlabel("S")
    plt.ylabel("position")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bellman_vs_deltas_t0.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    near = errors[errors["near_barrier"]]
    for strategy, df in near.groupby("strategy"):
        plt.hist(df["error"], bins=45, alpha=0.45, label=strategy)
    plt.xlabel("terminal hedge error")
    plt.ylabel("count")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "terminal_errors_near_barrier.png", dpi=150)
    plt.close()

    if not holds.empty:
        plt.figure(figsize=(7, 5))
        h = holds.groupby("lambda")[["near_barrier_hold", "far_hold"]].mean().reset_index()
        plt.plot(h["lambda"], h["near_barrier_hold"], marker="o", label="near barrier")
        plt.plot(h["lambda"], h["far_hold"], marker="s", label="far")
        plt.xlabel("lambda")
        plt.ylabel("HOLD fraction")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "hold_near_barrier_vs_far.png", dpi=150)
        plt.close()

    plt.figure(figsize=(7, 5))
    for strategy, df in metrics.groupby("strategy"):
        plt.scatter(df["rmse"], df["mean_cost"], label=strategy)
    plt.xlabel("RMSE")
    plt.ylabel("mean transaction cost")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cost_vs_rmse.png", dpi=150)
    plt.close()

    for (lam, path), df in trajectories.groupby(["lambda", "path"]):
        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        ax.plot(df["t"], df["S"], label="S")
        ax.axhline(BARRIER, color="black", linestyle="--", linewidth=1)
        ax2 = ax.twinx()
        ax2.plot(df["t"], df["q_bellman"], color="tab:orange", label="Bellman q")
        ax2.plot(df["t"], df["barrier_delta"], color="tab:green", linestyle=":", label="barrier delta")
        ax.set_xlabel("t")
        ax.set_ylabel("S")
        ax2.set_ylabel("position")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"trajectory_lambda_{lam:g}_path_{path}.png", dpi=150)
        plt.close()


def run_experiment():
    ensure_dirs()
    started = time.perf_counter()
    price_bench = barrier_benchmark_tree(n_steps=80, n_s=351, n_quad=9)
    initial_price = float(interp1(price_bench["values"][0], price_bench["s_grid"], S0))

    lambda0 = solve_lambda0_reduced(n_steps=24, n_s=201, n_quad=7, drift=MU)
    conv_t, conv_s = run_convergence(initial_price)
    cost_solutions = [solve_bellman_costs(lam, n_steps=4, grids=make_cost_grids(n_s=15, n_b=17, n_q=9), n_quad=3, drift=MU) for lam in [0.0005, 0.002, 0.01]]
    metrics, errors, crossing, trajectories = run_backtests(lambda0, cost_solutions, initial_price)
    holds = pd.concat([classify_buy_hold_sell(sol) for sol in cost_solutions], ignore_index=True)
    profile_trees = {24: barrier_benchmark_tree(n_steps=24, n_s=201, n_quad=7), 4: barrier_benchmark_tree(n_steps=4, n_s=201, n_quad=7)}
    profiles = policy_profiles(lambda0, cost_solutions, profile_trees)

    conv_t.to_csv(RESULT_DIR / "convergence_time.csv", index=False)
    conv_s.to_csv(RESULT_DIR / "convergence_S.csv", index=False)
    crossing.to_csv(RESULT_DIR / "barrier_crossing_diagnostics.csv", index=False)
    profiles.to_csv(RESULT_DIR / "policy_profiles.csv", index=False)
    metrics.to_csv(RESULT_DIR / "metrics.csv", index=False)
    metrics.to_csv(RESULT_DIR / "backtest_comparison.csv", index=False)
    errors.to_csv(RESULT_DIR / "terminal_error_samples.csv", index=False)
    holds.to_csv(RESULT_DIR / "buy_hold_sell_policy.csv", index=False)
    trajectories.to_csv(RESULT_DIR / "example_trajectories.csv", index=False)
    plot_outputs(metrics, profiles, errors, crossing, holds, trajectories)

    bell0 = metrics[(metrics["lambda"].eq(0.0)) & (metrics["strategy"].eq("bellman_lambda0"))].iloc[0].to_dict()
    bd0 = metrics[(metrics["lambda"].eq(0.0)) & (metrics["strategy"].eq("barrier_delta"))].iloc[0].to_dict()
    ed0 = metrics[(metrics["lambda"].eq(0.0)) & (metrics["strategy"].eq("european_delta"))].iloc[0].to_dict()
    hold_summary = holds.groupby("lambda")[["hold_fraction", "near_barrier_hold", "far_hold"]].mean().reset_index()
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "results_dir": str(RESULT_DIR),
        "initial_barrier_price_risk_neutral_numerical": initial_price,
        "formulation": {
            "state_lambda0": "(S,W,I), with exact quadratic reduction in W",
            "state_costs": "(S,b,q,I), explicit grid",
            "control": "new stock position a=q_prime",
            "objective": "E[(W_T - (S_T-K)^+ I_T)^2]; costs are deducted from cash",
            "barrier": "Brownian-bridge survival probability in Bellman and backtests",
            "measure_note": "Bellman/backtests use real drift mu=0.08; price/delta benchmarks use risk-neutral drift r=0.02.",
        },
        "lambda0_backtest": {"bellman": bell0, "barrier_delta": bd0, "european_delta": ed0},
        "crossing_diagnostics": crossing.to_dict(orient="records"),
        "hold_summary_costs": hold_summary.to_dict(orient="records"),
        "n_figures": len(list(FIG_DIR.glob("*.png"))),
    }
    write_json(RESULT_DIR / "summary.json", summary)
    write_json(RESULT_DIR / "config.json", {
        "script": Path(__file__).name,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "model": {"S0": S0, "K": K, "B": BARRIER, "T": T, "r": R, "mu": MU, "sigma": SIGMA},
        "lambda0": {"N": 24, "Ns": 201, "n_quad": 7},
        "costs": {"N": 4, "Ns": 15, "Nb": 17, "Nq": 9, "n_quad": 3, "lambdas": [0.0005, 0.002, 0.01]},
        "backtest": {"n_paths": N_PATHS, "n_fine_steps": N_FINE, "seed": SEED},
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_experiment()
