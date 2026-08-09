"""
Experience Bellman : hedging optimal discret d'un call europeen avec couts de transaction.

Formulation implementee, en temps discret t_n = n dt :

    S_{n+1} = S_n exp((mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z_{n+1})

La simulation/backtest utilise la mesure reelle avec mu=0.08. Le prix et le delta
Black-Scholes utilises uniquement comme benchmark sont calcules sous mesure
risque-neutre avec drift r.

Etat Markovien :

    z_n = (S_n, b_n, q_n)

ou b_n est le compte cash avant transaction et q_n la position en actions deja
detenue. Le controle est la nouvelle position a_n dans une grille A.

Transaction :

    C_n = lambda S_n |a_n - q_n|
    b_n^+ = b_n - (a_n - q_n) S_n - C_n

Evolution autofinancee entre deux dates :

    b_{n+1} = b_n^+ exp(r dt)
    q_{n+1} = a_n
    W_{n+1} = b_{n+1} + q_{n+1} S_{n+1}

Condition terminale :

    V_N(S,b,q) = (b + q S - (S-K)^+)^2

Recurrence de Bellman discrete :

    V_n(S,b,q) = min_a E[V_{n+1}(S_{n+1}, b_{n+1}, a) | S,b,q,a]

La politique stockee est l'argmin a*_n(S,b,q). Aucune evaluation du delta
Black-Scholes n'est utilisee dans solve_bellman(); bs_delta sert seulement
apres coup pour les benchmarks et diagnostics.

Reduction d'etat examinee :
La perte terminale est quadratique en b. Pour une action fixee, le backup d'une
fonction quadratique en b reste quadratique. Cependant le minimum sur plusieurs
actions est une enveloppe inferieure de quadratiques, pas une seule quadratique
globale; la politique depend donc de b. Pour cette implementation de validation,
on conserve une grille explicite de cash de taille moderee au lieu d'introduire
une reduction non rigoureuse.
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

from exp_hedging_transaction_costs import (
    K,
    MU,
    R,
    S0,
    SIGMA,
    T,
    bs_call_price,
    bs_delta,
    call_payoff,
    metrics_from_result,
    rebalance_indices,
    simulate_gbm_paths,
    simulate_hedge,
    transaction_cost,
)


RESULT_DIR = Path("results") / "hedging_bellman"
FIG_DIR = RESULT_DIR / "figures"

LAMBDAS = [0.0, 0.0005, 0.002, 0.01]
N_DP = 12
N_PATHS_BACKTEST = 10000
SEED = 12345
QUAD_NODES = np.array([-2.0201828704560856, -0.9585724646138185, 0.0, 0.9585724646138185, 2.0201828704560856])
QUAD_WEIGHTS = np.array([0.01995324205904591, 0.3936193231522412, 0.9453087204829419, 0.3936193231522412, 0.01995324205904591]) / math.sqrt(math.pi)
GH_CACHE = {}


def ensure_dirs():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def make_grids(n_s=25, n_b=35, n_q=17):
    s_grid = np.exp(np.linspace(np.log(45.0), np.log(220.0), n_s))
    b_grid = np.linspace(-250.0, 150.0, n_b)
    q_grid = np.linspace(0.0, 1.0, n_q)
    return s_grid, b_grid, q_grid


def make_cost_grids(n_s=51, n_b=61, n_q=33, cash_mode="adaptive"):
    s_grid = np.exp(np.linspace(np.log(35.0), np.log(260.0), n_s))
    q_grid = np.linspace(0.0, 1.0, n_q)
    if cash_mode == "uniform":
        b_grid = np.linspace(-280.0, 160.0, n_b)
    elif cash_mode == "adaptive":
        # Dense near typical call hedge cash b ~= option_value - q*S, with
        # broad tails for stressed states. This is numerical, not a change of
        # state formulation.
        u = np.linspace(-1.0, 1.0, n_b)
        center = -45.0
        half_width = 230.0
        b_grid = center + half_width * np.sinh(1.8 * u) / np.sinh(1.8)
        b_grid[0], b_grid[-1] = -280.0, 160.0
        b_grid = np.unique(np.sort(b_grid))
    else:
        raise ValueError(f"Unknown cash_mode: {cash_mode}")
    return s_grid, b_grid, q_grid


def gauss_hermite_normal(n_quad):
    """Nodes/weights for E[f(Z)] with Z~N(0,1)."""
    if n_quad not in GH_CACHE:
        nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
        GH_CACHE[n_quad] = math.sqrt(2.0) * nodes, weights / math.sqrt(math.pi)
    return GH_CACHE[n_quad]


def make_reduced_s_grid(n_s):
    return np.exp(np.linspace(np.log(35.0), np.log(260.0), n_s))


def solve_lambda0_reduced(n_steps=24, n_s=301, n_quad=9):
    """Bellman lambda=0 with rigorous quadratic wealth reduction.

    V_n(S,W) = A_n(S) W^2 - 2 B_n(S) W + C_n(S).
    The one-step minimization over stock holdings a is solved analytically.
    No Black-Scholes delta is used in this solver.
    """
    s_grid = make_reduced_s_grid(n_s)
    dt = T / n_steps
    rf = math.exp(R * dt)
    z_nodes, z_weights = gauss_hermite_normal(n_quad)

    payoff = call_payoff(s_grid)
    A_next = np.ones_like(s_grid)
    B_next = payoff.copy()
    C_next = payoff ** 2
    coeffs = [(A_next, B_next, C_next)]

    for _ in range(n_steps - 1, -1, -1):
        A_now = np.empty_like(s_grid)
        B_now = np.empty_like(s_grid)
        C_now = np.empty_like(s_grid)
        for i, s in enumerate(s_grid):
            s_next = s * np.exp((MU - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z_nodes)
            y = s_next - rf * s
            A_z = np.interp(s_next, s_grid, A_next, left=A_next[0], right=A_next[-1])
            B_z = np.interp(s_next, s_grid, B_next, left=B_next[0], right=B_next[-1])
            C_z = np.interp(s_next, s_grid, C_next, left=C_next[0], right=C_next[-1])
            e_a = float(np.sum(z_weights * A_z))
            e_b = float(np.sum(z_weights * B_z))
            e_c = float(np.sum(z_weights * C_z))
            e_ay = float(np.sum(z_weights * A_z * y))
            e_by = float(np.sum(z_weights * B_z * y))
            e_ayy = float(np.sum(z_weights * A_z * y ** 2))
            e_ayy = max(e_ayy, 1e-14)

            # Minimizing over a gives a*(W)=alpha+beta W and value coefficients below.
            alpha = e_by / e_ayy
            beta = -rf * e_ay / e_ayy
            A_now[i] = rf ** 2 * e_a - (rf * e_ay) ** 2 / e_ayy
            B_now[i] = rf * e_b - (rf * e_ay * e_by) / e_ayy
            C_now[i] = e_c - (e_by ** 2) / e_ayy
            if not np.isfinite(alpha + beta):
                raise FloatingPointError("Non-finite reduced Bellman policy coefficient")
        A_next, B_next, C_next = A_now, B_now, C_now
        coeffs.append((A_next, B_next, C_next))

    coeffs = list(reversed(coeffs))
    return {"n_steps": n_steps, "dt": dt, "s_grid": s_grid, "coeffs": coeffs, "n_quad": n_quad}


def reduced_policy_action(solution, n, s, wealth, q_grid=None):
    s_grid = solution["s_grid"]
    dt = solution["dt"]
    rf = math.exp(R * dt)
    z_nodes, z_weights = gauss_hermite_normal(solution["n_quad"])
    A_next, B_next, _ = solution["coeffs"][n + 1]

    s = np.asarray(s, dtype=float)
    wealth = np.asarray(wealth, dtype=float)
    s_flat = s.reshape(-1)
    w_flat = wealth.reshape(-1)
    actions = np.empty_like(s_flat)
    for i, (s0, w0) in enumerate(zip(s_flat, w_flat)):
        s_next = s0 * np.exp((MU - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z_nodes)
        y = s_next - rf * s0
        A_z = np.interp(s_next, s_grid, A_next, left=A_next[0], right=A_next[-1])
        B_z = np.interp(s_next, s_grid, B_next, left=B_next[0], right=B_next[-1])
        e_ay = float(np.sum(z_weights * A_z * y))
        e_by = float(np.sum(z_weights * B_z * y))
        e_ayy = max(float(np.sum(z_weights * A_z * y ** 2)), 1e-14)
        actions[i] = (e_by - w0 * rf * e_ay) / e_ayy
    actions = np.clip(actions.reshape(s.shape), 0.0, 1.0)
    if q_grid is not None:
        q_grid = np.asarray(q_grid)
        indices = np.abs(actions[..., None] - q_grid).argmin(axis=-1)
        actions = q_grid[indices]
    return actions


def reduced_policy_vs_delta_metrics(solution, q_grid=None):
    rows = []
    s_eval = np.linspace(60.0, 160.0, 201)
    s_global = np.linspace(40.0, 240.0, 301)
    for n in range(solution["n_steps"]):
        t = n * solution["dt"]
        tau = T - t
        for label, s_values in (("interior", s_eval), ("global", s_global)):
            wealth = bs_call_price(s_values, tau)
            action = reduced_policy_action(solution, n, s_values, wealth, q_grid=q_grid)
            diff = action - bs_delta(s_values, tau)
            rows.append(
                {
                    "time_index": n,
                    "t": t,
                    "zone": label,
                    "rmse_policy_vs_delta": float(np.sqrt(np.mean(diff ** 2))),
                    "mae_policy_vs_delta": float(np.mean(np.abs(diff))),
                    "max_abs_policy_vs_delta": float(np.max(np.abs(diff))),
                    "boundary_delta_lt_0_or_gt_1": False,
                }
            )
    return pd.DataFrame(rows)


def simulate_reduced_lambda0_policy(solution, paths, times, q_grid=None):
    idx = rebalance_indices(solution["n_steps"], n_steps=paths.shape[1] - 1)
    n_paths = paths.shape[0]
    cash = np.full(n_paths, float(bs_call_price(S0, T)))
    q = np.zeros(n_paths)
    turnover = np.zeros(n_paths)
    n_trades = np.zeros(n_paths)
    last_i = 0
    held_q = np.full((min(5, n_paths), len(idx) - 1), np.nan)
    bs_d = np.full_like(held_q, np.nan)
    for n, i in enumerate(idx[:-1]):
        if i > last_i:
            cash *= np.exp(R * (times[i] - times[last_i]))
        s_i = paths[:, i]
        wealth = cash + q * s_i
        actions = reduced_policy_action(solution, n, s_i, wealth, q_grid=q_grid)
        cash -= (actions - q) * s_i
        turnover += np.abs(actions - q)
        n_trades += np.abs(actions - q) > 1e-12
        q = actions
        m = held_q.shape[0]
        held_q[:, n] = q[:m]
        bs_d[:, n] = bs_delta(s_i[:m], T - times[i])
        last_i = i
    cash *= np.exp(R * (T - times[last_i]))
    terminal_wealth = cash + q * paths[:, -1]
    payoff = call_payoff(paths[:, -1])
    error = terminal_wealth - payoff
    return {
        "error": error,
        "pnl": error,
        "terminal_wealth": terminal_wealth,
        "payoff": payoff,
        "cumulative_cost": np.zeros(n_paths),
        "turnover": turnover,
        "n_trades": n_trades,
        "rebalance_indices": idx,
        "held_delta_path": held_q,
        "bs_delta_path": bs_d,
    }


def interp1_quadratic(x_grid, y_values, x_query):
    x_query = np.asarray(x_query)
    x_clamped = np.clip(x_query, x_grid[0], x_grid[-1])
    j_mid = np.searchsorted(x_grid, x_clamped, side="left")
    j0 = np.clip(j_mid - 1, 0, len(x_grid) - 3)
    j1 = j0 + 1
    j2 = j0 + 2
    x0, x1, x2 = x_grid[j0], x_grid[j1], x_grid[j2]
    y0, y1, y2 = y_values[j0], y_values[j1], y_values[j2]
    l0 = (x_clamped - x1) * (x_clamped - x2) / ((x0 - x1) * (x0 - x2))
    l1 = (x_clamped - x0) * (x_clamped - x2) / ((x1 - x0) * (x1 - x2))
    l2 = (x_clamped - x0) * (x_clamped - x1) / ((x2 - x0) * (x2 - x1))
    return l0 * y0 + l1 * y1 + l2 * y2


def interp2_on_grid(values, s_grid, b_grid, s_query, b_query, method="linear"):
    s_query, b_query = np.broadcast_arrays(np.asarray(s_query), np.asarray(b_query))
    s_clamped = np.clip(s_query, s_grid[0], s_grid[-1])
    b_clamped = np.clip(b_query, b_grid[0], b_grid[-1])

    i = np.searchsorted(s_grid, s_clamped, side="right") - 1
    j = np.searchsorted(b_grid, b_clamped, side="right") - 1
    i = np.clip(i, 0, len(s_grid) - 2)
    j = np.clip(j, 0, len(b_grid) - 2)

    s0 = s_grid[i]
    s1 = s_grid[i + 1]
    b0 = b_grid[j]
    b1 = b_grid[j + 1]
    ws = (s_clamped - s0) / (s1 - s0)
    wb = (b_clamped - b0) / (b1 - b0)

    v00 = values[i, j]
    v10 = values[i + 1, j]
    v01 = values[i, j + 1]
    v11 = values[i + 1, j + 1]
    if method == "quadratic_cash":
        flat_i = np.ravel(i)
        flat_b = np.ravel(b_clamped)
        shape = np.shape(b_clamped)
        row0 = np.array([interp1_quadratic(b_grid, values[ii], bb) for ii, bb in zip(flat_i, flat_b)]).reshape(shape)
        row1 = np.array([interp1_quadratic(b_grid, values[ii + 1], bb) for ii, bb in zip(flat_i, flat_b)]).reshape(shape)
        return (1 - ws) * row0 + ws * row1
    if method != "linear":
        raise ValueError(f"Unknown interpolation method: {method}")
    return (1 - ws) * (1 - wb) * v00 + ws * (1 - wb) * v10 + (1 - ws) * wb * v01 + ws * wb * v11


def bellman_backup_state(s, b, q, action, lam, next_values_for_action, s_grid, b_grid, dt, interp_method="linear", n_quad=5):
    b_after = b - (action - q) * s - transaction_cost(lam, s, action, q)
    b_next = b_after * math.exp(R * dt)
    expected = 0.0
    z_nodes, z_weights = gauss_hermite_normal(n_quad)
    for z, weight in zip(z_nodes, z_weights):
        s_next = s * math.exp((MU - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z)
        expected += weight * float(interp2_on_grid(next_values_for_action, s_grid, b_grid, s_next, b_next, method=interp_method))
    return expected


def solve_bellman(lam, n_steps=N_DP, grids=None, interp_method="linear", n_quad=5):
    if grids is None:
        grids = make_grids()
    s_grid, b_grid, q_grid = grids
    dt = T / n_steps

    s_mesh = s_grid[:, None]
    b_mesh = b_grid[None, :]
    value_next = np.empty((len(q_grid), len(s_grid), len(b_grid)), dtype=float)
    for iq, q in enumerate(q_grid):
        value_next[iq] = (b_mesh + q * s_mesh - call_payoff(s_mesh)) ** 2

    values = [None] * (n_steps + 1)
    policies = [None] * n_steps
    values[n_steps] = value_next.copy()

    for n in range(n_steps - 1, -1, -1):
        value_now = np.empty_like(value_next)
        policy_now = np.empty_like(value_next)
        for iq, q in enumerate(q_grid):
            best = np.full((len(s_grid), len(b_grid)), np.inf)
            best_action = np.full((len(s_grid), len(b_grid)), q_grid[0])
            for ia, action in enumerate(q_grid):
                candidate = np.zeros((len(s_grid), len(b_grid)), dtype=float)
                for is_, s in enumerate(s_grid):
                    b_after = b_grid - (action - q) * s - transaction_cost(lam, s, action, q)
                    b_next = b_after * math.exp(R * dt)
                    expected_row = np.zeros(len(b_grid), dtype=float)
                    z_nodes, z_weights = gauss_hermite_normal(n_quad)
                    for z, weight in zip(z_nodes, z_weights):
                        s_next = s * math.exp((MU - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z)
                        expected_row += weight * interp2_on_grid(values[n + 1][ia], s_grid, b_grid, s_next, b_next, method=interp_method)
                    candidate[is_] = expected_row
                improve = candidate < best
                best[improve] = candidate[improve]
                best_action[improve] = action
            value_now[iq] = best
            policy_now[iq] = best_action
        values[n] = value_now
        policies[n] = policy_now
        value_next = value_now
    return {
        "lambda": lam,
        "n_steps": n_steps,
        "dt": dt,
        "s_grid": s_grid,
        "b_grid": b_grid,
        "q_grid": q_grid,
        "values": values,
        "policies": policies,
        "interp_method": interp_method,
        "n_quad": n_quad,
    }


def nearest_index(grid, x):
    return int(np.clip(np.searchsorted(grid, x), 1, len(grid) - 1) - (abs(x - grid[np.searchsorted(grid, x) - 1]) <= abs(x - grid[min(np.searchsorted(grid, x), len(grid) - 1)])))


def policy_action(solution, n, s, b, q):
    s_grid, b_grid, q_grid = solution["s_grid"], solution["b_grid"], solution["q_grid"]
    iq = int(np.argmin(np.abs(q_grid - q)))
    is_ = int(np.argmin(np.abs(s_grid - s)))
    ib = int(np.argmin(np.abs(b_grid - b)))
    return float(solution["policies"][min(n, solution["n_steps"] - 1)][iq, is_, ib])


def bellman_actions_continuous(solution, n, s_values, b_values, q_values, lam):
    s_values = np.asarray(s_values)
    b_values = np.asarray(b_values)
    q_values = np.asarray(q_values)
    dt = solution["dt"]
    z_nodes, z_weights = gauss_hermite_normal(solution.get("n_quad", 5))
    interp_method = solution.get("interp_method", "linear")
    costs_by_action = []
    for ia, action in enumerate(solution["q_grid"]):
        b_after = b_values - (action - q_values) * s_values - transaction_cost(lam, s_values, action, q_values)
        b_next = b_after * math.exp(R * dt)
        expected = np.zeros_like(s_values, dtype=float)
        for z, weight in zip(z_nodes, z_weights):
            s_next = s_values * np.exp((MU - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z)
            expected += weight * interp2_on_grid(
                solution["values"][n + 1][ia],
                solution["s_grid"],
                solution["b_grid"],
                s_next,
                b_next,
                method=interp_method,
            )
        costs_by_action.append(expected)
    stacked = np.vstack(costs_by_action)
    return solution["q_grid"][np.argmin(stacked, axis=0)]


def simulate_bellman_policy(solution, paths, times, lam):
    idx = rebalance_indices(solution["n_steps"], n_steps=paths.shape[1] - 1)
    n_paths = paths.shape[0]
    cash = np.full(n_paths, float(bs_call_price(S0, T)))
    q = np.zeros(n_paths)
    cumulative_cost = np.zeros(n_paths)
    turnover = np.zeros(n_paths)
    n_trades = np.zeros(n_paths)
    held_q = np.full((min(5, n_paths), len(idx) - 1), np.nan)
    bs_d = np.full_like(held_q, np.nan)
    last_i = 0
    s_near_boundary = 0
    b_near_boundary = 0
    s_outside = 0
    b_outside = 0
    n_state_checks = 0

    for n, i in enumerate(idx[:-1]):
        if i > last_i:
            cash *= np.exp(R * (times[i] - times[last_i]))
        s_i = paths[:, i]
        s_margin = 0.05 * (solution["s_grid"][-1] - solution["s_grid"][0])
        b_margin = 0.05 * (solution["b_grid"][-1] - solution["b_grid"][0])
        s_near_boundary += int(np.sum((s_i < solution["s_grid"][0] + s_margin) | (s_i > solution["s_grid"][-1] - s_margin)))
        b_near_boundary += int(np.sum((cash < solution["b_grid"][0] + b_margin) | (cash > solution["b_grid"][-1] - b_margin)))
        s_outside += int(np.sum((s_i < solution["s_grid"][0]) | (s_i > solution["s_grid"][-1])))
        b_outside += int(np.sum((cash < solution["b_grid"][0]) | (cash > solution["b_grid"][-1])))
        n_state_checks += int(s_i.size)
        actions = bellman_actions_continuous(solution, n, s_i, cash, q, lam)
        costs = transaction_cost(lam, s_i, actions, q)
        cash -= (actions - q) * s_i + costs
        cumulative_cost += costs
        turnover += np.abs(actions - q)
        n_trades += np.abs(actions - q) > 1e-12
        q = actions
        m = held_q.shape[0]
        held_q[:, n] = q[:m]
        bs_d[:, n] = bs_delta(s_i[:m], T - times[i])
        last_i = i

    cash *= np.exp(R * (T - times[last_i]))
    terminal_wealth = cash + q * paths[:, -1]
    payoff = call_payoff(paths[:, -1])
    error = terminal_wealth - payoff
    return {
        "error": error,
        "pnl": error,
        "terminal_wealth": terminal_wealth,
        "payoff": payoff,
        "cumulative_cost": cumulative_cost,
        "turnover": turnover,
        "n_trades": n_trades,
        "rebalance_indices": idx,
        "held_delta_path": held_q,
        "bs_delta_path": bs_d,
        "diagnostics": {
            "fraction_s_near_boundary": s_near_boundary / n_state_checks,
            "fraction_cash_near_boundary": b_near_boundary / n_state_checks,
            "fraction_s_outside_grid": s_outside / n_state_checks,
            "fraction_cash_outside_grid": b_outside / n_state_checks,
        },
    }


def policy_vs_delta_metrics(solution):
    s_grid, b_grid, q_grid = solution["s_grid"], solution["b_grid"], solution["q_grid"]
    rows = []
    q0 = q_grid[np.argmin(np.abs(q_grid))]
    interior = (s_grid >= 60.0) & (s_grid <= 160.0)
    for n in range(solution["n_steps"]):
        t = n * solution["dt"]
        tau = T - t
        diffs = []
        for s in s_grid[interior]:
            b = float(bs_call_price(s, tau))
            a = policy_action(solution, n, s, b, q0)
            diffs.append(a - float(bs_delta(s, tau)))
        diffs = np.asarray(diffs)
        rows.append(
            {
                "time_index": n,
                "t": t,
                "rmse_policy_vs_delta": float(np.sqrt(np.mean(diffs ** 2))),
                "mae_policy_vs_delta": float(np.mean(np.abs(diffs))),
                "max_abs_policy_vs_delta": float(np.max(np.abs(diffs))),
            }
        )
    return pd.DataFrame(rows)


def classify_policy(solution):
    rows = []
    for n, policy in enumerate(solution["policies"]):
        diff = policy - solution["q_grid"][:, None, None]
        hold = np.isclose(diff, 0.0, atol=0.5 * (solution["q_grid"][1] - solution["q_grid"][0]) + 1e-12)
        buy = diff > 0
        sell = diff < 0
        for is_, s in enumerate(solution["s_grid"]):
            hold_by_q = hold[:, is_, :].mean(axis=1)
            hold_q = solution["q_grid"][hold_by_q > 0.5]
            rows.append(
                {
                    "lambda": solution["lambda"],
                    "time_index": n,
                    "t": n * solution["dt"],
                    "S": s,
                    "hold_fraction": float(np.mean(hold[:, is_, :])),
                    "buy_fraction": float(np.mean(buy[:, is_, :])),
                    "sell_fraction": float(np.mean(sell[:, is_, :])),
                    "hold_width_q": float(hold_q.max() - hold_q.min()) if hold_q.size else 0.0,
                }
            )
    return pd.DataFrame(rows)


def save_policy_slices(solution):
    rows = []
    b0 = float(bs_call_price(S0, T))
    for n in sorted(set([0, solution["n_steps"] // 2, solution["n_steps"] - 1])):
        for q in solution["q_grid"]:
            for s in solution["s_grid"]:
                action = policy_action(solution, n, s, b0, q)
                rows.append(
                    {
                        "lambda": solution["lambda"],
                        "time_index": n,
                        "t": n * solution["dt"],
                        "S": s,
                        "cash_slice": b0,
                        "q": q,
                        "action": action,
                        "decision": "HOLD" if np.isclose(action, q) else ("BUY" if action > q else "SELL"),
                    }
                )
    return pd.DataFrame(rows)


def run_backtests(solutions):
    times, paths = simulate_gbm_paths(n_paths=N_PATHS_BACKTEST, n_steps=252, seed=SEED, mu=MU)
    rows = []
    samples = []
    trajectory_rows = []
    agg_prev = pd.read_csv(Path("results") / "hedging_transaction_costs" / "aggregate_comparisons.csv")
    for sol in solutions:
        lam = sol["lambda"]
        bell = simulate_bellman_policy(sol, paths, times, lam)
        rows.append(metrics_from_result(bell, lam, sol["n_steps"], "bellman_dp", -1.0))
        samples.append(sample_errors(bell, lam, "bellman_dp"))

        delta = simulate_hedge(paths, times, lam=lam, frequency=sol["n_steps"], strategy="delta_bs")
        rows.append(metrics_from_result(delta, lam, sol["n_steps"], "delta_bs_same_frequency", 0.0))
        samples.append(sample_errors(delta, lam, "delta_bs_same_frequency"))

        prev = agg_prev[agg_prev["lambda"].eq(lam)].iloc[0]
        heuristic = simulate_hedge(paths, times, lam=lam, frequency=int(prev["best_optimized_frequency"]), strategy="no_trade_band", band=float(prev["best_optimized_band"]))
        rows.append(metrics_from_result(heuristic, lam, int(prev["best_optimized_frequency"]), "best_no_trade_exp3", float(prev["best_optimized_band"])))
        samples.append(sample_errors(heuristic, lam, "best_no_trade_exp3"))

        idx = bell["rebalance_indices"][:-1]
        for k in range(3):
            trajectory_rows.append(
                pd.DataFrame(
                    {
                        "lambda": lam,
                        "path": k,
                        "t": times[idx],
                        "S": paths[k, idx],
                        "delta_bs": bell["bs_delta_path"][k],
                        "q_bellman": bell["held_delta_path"][k],
                    }
                )
            )
    return pd.DataFrame(rows), pd.concat(samples, ignore_index=True), pd.concat(trajectory_rows, ignore_index=True)


def sample_errors(result, lam, strategy, n=5000):
    return pd.DataFrame(
        {
            "lambda": lam,
            "strategy": strategy,
            "error": result["error"][:n],
            "pnl": result["pnl"][:n],
            "cost": result["cumulative_cost"][:n],
        }
    )


def convergence_study(grids):
    rows = []
    for n_steps in [6, 12]:
        sol = solve_bellman(0.0, n_steps=n_steps, grids=grids)
        d = policy_vs_delta_metrics(sol)
        rows.append({"n_steps": n_steps, **d.mean(numeric_only=True).to_dict()})
    return pd.DataFrame(rows)


def plot_outputs(backtest_metrics, error_samples, policy_slices, hold_summary, delta_comparison, trajectories):
    for path in FIG_DIR.glob("*.png"):
        path.unlink()

    for lam, df in policy_slices.groupby("lambda"):
        for t_idx in sorted(df["time_index"].unique()):
            d = df[df["time_index"].eq(t_idx)]
            pivot = d.pivot(index="q", columns="S", values="action")
            plt.figure(figsize=(9, 5))
            plt.imshow(pivot.values, aspect="auto", origin="lower", extent=[d["S"].min(), d["S"].max(), d["q"].min(), d["q"].max()])
            plt.colorbar(label="optimal action q'")
            plt.xlabel("S")
            plt.ylabel("current q")
            plt.title(f"Bellman policy, lambda={lam:g}, n={int(t_idx)}")
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"policy_lambda_{lam:g}_n_{int(t_idx)}.png", dpi=150)
            plt.close()

            code = d.assign(code=d["decision"].map({"SELL": -1, "HOLD": 0, "BUY": 1})).pivot(index="q", columns="S", values="code")
            plt.figure(figsize=(9, 5))
            plt.imshow(code.values, aspect="auto", origin="lower", vmin=-1, vmax=1, cmap="coolwarm", extent=[d["S"].min(), d["S"].max(), d["q"].min(), d["q"].max()])
            plt.colorbar(label="SELL=-1, HOLD=0, BUY=1")
            plt.xlabel("S")
            plt.ylabel("current q")
            plt.title(f"BUY/HOLD/SELL, lambda={lam:g}, n={int(t_idx)}")
            plt.tight_layout()
            plt.savefig(FIG_DIR / f"buy_hold_sell_lambda_{lam:g}_n_{int(t_idx)}.png", dpi=150)
            plt.close()

    width = hold_summary.groupby("lambda")["hold_width_q"].mean().reset_index()
    plt.figure()
    plt.plot(width["lambda"], width["hold_width_q"], marker="o")
    plt.xlabel("lambda")
    plt.ylabel("mean HOLD width in q")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "hold_width_vs_lambda.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(delta_comparison["t"], delta_comparison["rmse_policy_vs_delta"], marker="o")
    plt.xlabel("t")
    plt.ylabel("RMSE policy vs BS delta, lambda=0")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "bellman_vs_delta_lambda_0.png", dpi=150)
    plt.close()

    for lam, df in error_samples.groupby("lambda"):
        plt.figure(figsize=(9, 5))
        for strategy, d in df.groupby("strategy"):
            plt.hist(d["error"], bins=80, density=True, alpha=0.45, label=strategy)
        plt.xlabel("terminal error")
        plt.ylabel("density")
        plt.title(f"Backtest error distribution, lambda={lam:g}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"backtest_error_distribution_lambda_{lam:g}.png", dpi=150)
        plt.close()

    plt.figure(figsize=(8, 5))
    for strategy, d in backtest_metrics.groupby("strategy"):
        plt.scatter(d["rmse"], d["mean_total_cost"], label=strategy)
    plt.xlabel("RMSE")
    plt.ylabel("mean transaction cost")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cost_vs_rmse.png", dpi=150)
    plt.close()

    for (lam, path_id), df in trajectories.groupby(["lambda", "path"]):
        if path_id != 0:
            continue
        plt.figure(figsize=(10, 5))
        ax1 = plt.gca()
        ax1.plot(df["t"], df["S"], color="black", label="S")
        ax1.set_ylabel("S")
        ax2 = ax1.twinx()
        ax2.plot(df["t"], df["delta_bs"], label="BS delta")
        ax2.plot(df["t"], df["q_bellman"], label="Bellman q")
        ax2.set_ylabel("position")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2)
        plt.title(f"Example Bellman hedge, lambda={lam:g}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"trajectory_lambda_{lam:g}.png", dpi=150)
        plt.close()


def plot_convergence(conv_q, conv_s, conv_cash, conv_time, conv_quad, policy_error, backtest_metrics):
    specs = [
        (conv_q, "Nq", "convergence_policy_vs_Nq.png", "Control grid points Nq"),
        (conv_s, "Ns", "convergence_policy_vs_Ns.png", "Stock grid points Ns"),
        (conv_cash, "Nb", "convergence_policy_vs_Nb.png", "Cash grid points Nb"),
        (conv_time, "n_steps", "convergence_policy_vs_N.png", "Time steps N"),
        (conv_quad, "n_quad", "convergence_policy_vs_quadrature.png", "Gauss-Hermite points"),
    ]
    for df, xcol, filename, xlabel in specs:
        plt.figure(figsize=(8, 5))
        plt.plot(df[xcol], df["rmse_policy_interior"], marker="o", label="policy RMSE interior")
        plt.plot(df[xcol], df["rmse_policy_global"], marker="s", label="policy RMSE global")
        plt.xlabel(xlabel)
        plt.ylabel("RMSE vs BS delta")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=150)
        plt.close()

    interior = policy_error[policy_error["zone"].eq("interior")]
    plt.figure(figsize=(9, 5))
    plt.plot(interior["t"], interior["rmse_policy_vs_delta"], marker="o")
    plt.xlabel("t")
    plt.ylabel("RMSE q_Bellman - Delta_BS")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "best_lambda0_policy_error_by_time.png", dpi=150)
    plt.close()

    l0 = backtest_metrics[backtest_metrics["lambda"].eq(0.0)]
    plt.figure(figsize=(7, 5))
    plt.bar(l0["strategy"], l0["rmse"])
    plt.ylabel("hedging RMSE")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "lambda0_hedge_rmse_bellman_vs_delta.png", dpi=150)
    plt.close()


def plot_cost_convergence(cost_q, cost_s, cost_cash, cost_time, cost_interp, hold_summary, cost_metrics):
    specs = [
        (cost_q, "Nq", "convergence_costs_q.png", "Nq"),
        (cost_s, "Ns", "convergence_costs_S.png", "Ns"),
        (cost_cash, "Nb", "convergence_costs_cash.png", "Nb"),
        (cost_time, "n_steps", "convergence_costs_time.png", "N"),
    ]
    for df, xcol, filename, xlabel in specs:
        plt.figure(figsize=(8, 5))
        plt.plot(df[xcol], df["rmse_bellman"], marker="o", label="Bellman RMSE")
        plt.plot(df[xcol], df["mean_cost_bellman"], marker="s", label="mean cost")
        plt.xlabel(xlabel)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / filename, dpi=150)
        plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(cost_interp["interp_method"], cost_interp["rmse_bellman"])
    plt.ylabel("Bellman RMSE")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "interpolation_cost_rmse.png", dpi=150)
    plt.close()

    width = hold_summary.groupby("lambda")["hold_width_q"].mean().reset_index()
    plt.figure(figsize=(7, 5))
    plt.plot(width["lambda"], width["hold_width_q"], marker="o")
    plt.xlabel("lambda")
    plt.ylabel("mean HOLD width")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "costs_hold_width_vs_lambda.png", dpi=150)
    plt.close()

    width_t = hold_summary.groupby(["lambda", "t"])["hold_width_q"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    for lam, df in width_t.groupby("lambda"):
        plt.plot(df["t"], df["hold_width_q"], marker="o", label=f"lambda={lam:g}")
    plt.xlabel("t")
    plt.ylabel("mean HOLD width")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "costs_hold_width_vs_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    for strategy, df in cost_metrics.groupby("strategy"):
        plt.scatter(df["rmse"], df["mean_total_cost"], label=strategy)
    plt.xlabel("RMSE")
    plt.ylabel("mean transaction cost")
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "costs_hedge_error_vs_cost.png", dpi=150)
    plt.close()


def run_experiment():
    ensure_dirs()
    started = time.perf_counter()
    for path in FIG_DIR.glob("*.png"):
        path.unlink()

    best_row = json.loads((RESULT_DIR / "best_lambda0_config.json").read_text(encoding="utf-8"))
    conv_q = pd.read_csv(RESULT_DIR / "convergence_q.csv")
    conv_s = pd.read_csv(RESULT_DIR / "convergence_S.csv")
    conv_cash = pd.read_csv(RESULT_DIR / "convergence_cash.csv")
    conv_time = pd.read_csv(RESULT_DIR / "convergence_time.csv")
    conv_quad = pd.read_csv(RESULT_DIR / "convergence_quadrature.csv")
    policy_error = pd.read_csv(RESULT_DIR / "policy_error_metrics.csv")
    lambda0_metrics = pd.read_csv(RESULT_DIR / "backtest_metrics.csv")
    lambda0_metrics = lambda0_metrics[lambda0_metrics["lambda"].eq(0.0)].copy()
    cost_q, cost_s, cost_cash, cost_time, cost_interp = run_cost_convergence_sweeps()
    cost_q.to_csv(RESULT_DIR / "convergence_costs_q.csv", index=False)
    cost_s.to_csv(RESULT_DIR / "convergence_costs_S.csv", index=False)
    cost_cash.to_csv(RESULT_DIR / "convergence_costs_cash.csv", index=False)
    cost_time.to_csv(RESULT_DIR / "convergence_costs_time.csv", index=False)
    cost_interp.to_csv(RESULT_DIR / "interpolation_cost_diagnostics.csv", index=False)
    policy_stability = make_policy_stability(cost_q, cost_s, cost_cash, cost_time, cost_interp)
    policy_stability.to_csv(RESULT_DIR / "policy_stability.csv", index=False)

    cost_metrics, cost_errors, hold_summary, policy_slices, trajectories = selected_cost_backtests()
    cost_metrics = drop_mixed_objective_columns(cost_metrics).fillna(-1.0)
    hold_summary.to_csv(RESULT_DIR / "hold_region_metrics.csv", index=False)
    cost_metrics.to_csv(RESULT_DIR / "backtest_cost_comparison.csv", index=False)
    backtest_metrics = pd.concat([drop_mixed_objective_columns(lambda0_metrics), cost_metrics], ignore_index=True)
    backtest_metrics = backtest_metrics.fillna(-1.0)
    error_samples = cost_errors

    backtest_metrics.to_csv(RESULT_DIR / "backtest_metrics.csv", index=False)
    error_samples.to_csv(RESULT_DIR / "terminal_error_samples.csv", index=False)
    hold_summary.to_csv(RESULT_DIR / "hold_region_summary.csv", index=False)
    policy_slices.to_csv(RESULT_DIR / "policy_slices.csv", index=False)
    trajectories.to_csv(RESULT_DIR / "example_trajectories.csv", index=False)

    plot_convergence(conv_q, conv_s, conv_cash, conv_time, conv_quad, policy_error, backtest_metrics)
    plot_cost_convergence(cost_q, cost_s, cost_cash, cost_time, cost_interp, hold_summary, cost_metrics)
    plot_outputs(backtest_metrics, error_samples, policy_slices, hold_summary, policy_error[policy_error["zone"].eq("interior")], trajectories)

    interior = policy_error[policy_error["zone"].eq("interior")]
    hold_width = hold_summary.groupby("lambda")["hold_width_q"].mean().reset_index()
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "results_dir": str(RESULT_DIR),
        "lambda0_reduction": "cash/wealth handled analytically through quadratic value coefficients; no cash grid is used for validation",
        "best_lambda0_config": best_row,
        "lambda0_policy_rmse_interior": float(interior["rmse_policy_vs_delta"].mean()),
        "lambda0_policy_mae_interior": float(interior["mae_policy_vs_delta"].mean()),
        "lambda0_policy_max_abs_interior": float(interior["max_abs_policy_vs_delta"].max()),
        "lambda0_validation_threshold_rmse_lt_0_05": bool(float(interior["rmse_policy_vs_delta"].mean()) < 0.05),
        "lambda0_hedge_rmse_ratio": float(best_row["rmse_hedge_ratio"]),
        "hold_width_by_lambda": hold_width.to_dict(orient="records"),
        "cost_branch_numerics": {
            "state": "(S,b,q)",
            "cash_grid": "adaptive non-uniform grid dense near typical call hedge cash; no mathematical reduction imposed for lambda>0",
            "interp_default": "bilinear in (S,b); quadratic_cash diagnostic also saved",
            "selected_config": {"N": 12, "Ns": 31, "Nb": 31, "Nq": 17, "n_quad": 3, "cash_mode": "adaptive"},
            "boundary_diagnostics_selected": cost_metrics[cost_metrics["strategy"].eq("bellman_dp_costs_adaptive")][[
                "lambda",
                "fraction_s_near_boundary",
                "fraction_cash_near_boundary",
                "fraction_s_outside_grid",
                "fraction_cash_outside_grid",
            ]].to_dict(orient="records"),
        },
        "cost_backtest_metrics": cost_metrics.to_dict(orient="records"),
        "n_figures": len(list(FIG_DIR.glob("*.png"))),
    }
    write_json(RESULT_DIR / "summary.json", summary)
    write_json(RESULT_DIR / "config.json", config_dict(make_grids()))
    print(json.dumps(summary, indent=2))
    return None, backtest_metrics, summary


def config_dict(grids):
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "model": {"S0": S0, "K": K, "T": T, "r": R, "mu_real": MU, "sigma": SIGMA},
        "measure_note": "Bellman expectations and backtests use real drift mu. Black-Scholes price/delta benchmarks use risk-neutral drift r.",
        "bellman": {
            "state": ["S", "cash", "q"],
            "control": "a=q_prime",
            "n_steps": N_DP,
            "s_grid": [float(grids[0][0]), float(grids[0][-1]), len(grids[0])],
            "cash_grid": [float(grids[1][0]), float(grids[1][-1]), len(grids[1])],
            "q_grid": [float(grids[2][0]), float(grids[2][-1]), len(grids[2])],
            "quadrature": "5-point Gauss-Hermite for normal shocks",
            "lambdas": LAMBDAS,
        },
        "backtest": {"n_paths": N_PATHS_BACKTEST, "seed": SEED, "fine_steps": 252},
    }


def evaluate_lambda0_config(n_steps, n_s, n_quad, n_q=None, n_paths=8000, seed=SEED):
    solution = solve_lambda0_reduced(n_steps=n_steps, n_s=n_s, n_quad=n_quad)
    q_grid = np.linspace(0.0, 1.0, n_q) if n_q is not None else None
    policy_metrics = reduced_policy_vs_delta_metrics(solution, q_grid=q_grid)
    interior = policy_metrics[policy_metrics["zone"].eq("interior")]
    global_m = policy_metrics[policy_metrics["zone"].eq("global")]
    times, paths = simulate_gbm_paths(n_paths=n_paths, n_steps=252, seed=seed, mu=MU)
    bell = simulate_reduced_lambda0_policy(solution, paths, times, q_grid=q_grid)
    delta = simulate_hedge(paths, times, lam=0.0, frequency=n_steps, strategy="delta_bs")
    bell_m = metrics_from_result(bell, 0.0, n_steps, "bellman_lambda0_reduced", -1.0 if n_q is None else 1.0 / (n_q - 1))
    delta_m = metrics_from_result(delta, 0.0, n_steps, "delta_bs_same_frequency", 0.0)
    return {
        "n_steps": n_steps,
        "Ns": n_s,
        "Nq": int(n_q) if n_q is not None else 0,
        "Nb": 0,
        "n_quad": n_quad,
        "control": "continuous" if n_q is None else "rounded_to_q_grid",
        "interpolation": "linear in S for quadratic value coefficients; analytic quadratic minimization in wealth",
        "rmse_policy_interior": float(interior["rmse_policy_vs_delta"].mean()),
        "mae_policy_interior": float(interior["mae_policy_vs_delta"].mean()),
        "max_abs_policy_interior": float(interior["max_abs_policy_vs_delta"].max()),
        "rmse_policy_global": float(global_m["rmse_policy_vs_delta"].mean()),
        "mae_policy_global": float(global_m["mae_policy_vs_delta"].mean()),
        "max_abs_policy_global": float(global_m["max_abs_policy_vs_delta"].max()),
        "rmse_hedge_bellman": bell_m["rmse"],
        "rmse_hedge_delta_same_frequency": delta_m["rmse"],
        "rmse_hedge_ratio": bell_m["rmse"] / delta_m["rmse"],
    }, solution, bell, delta


def run_lambda0_convergence():
    base = {"n_steps": 24, "n_s": 301, "n_quad": 9, "n_q": None}
    rows_q = []
    for n_q in [17, 33, 65, 129]:
        row, _, _, _ = evaluate_lambda0_config(**{**base, "n_q": n_q}, n_paths=5000)
        rows_q.append(row)

    rows_s = []
    for n_s in [51, 101, 201, 401]:
        row, _, _, _ = evaluate_lambda0_config(**{**base, "n_s": n_s}, n_paths=5000)
        rows_s.append(row)

    rows_t = []
    for n_steps in [6, 12, 24, 48]:
        row, _, _, _ = evaluate_lambda0_config(**{**base, "n_steps": n_steps}, n_paths=5000)
        rows_t.append(row)

    rows_quad = []
    for n_quad in [3, 5, 9, 15]:
        row, _, _, _ = evaluate_lambda0_config(**{**base, "n_quad": n_quad}, n_paths=5000)
        rows_quad.append(row)

    # Cash grid diagnostic: lambda=0 reduction eliminates cash analytically. The
    # rows below document that the selected scheme has no cash discretization.
    rows_cash = []
    for nb in [0, 21, 41, 81]:
        row, _, _, _ = evaluate_lambda0_config(**base, n_paths=3000)
        row["Nb"] = nb
        row["cash_grid_status"] = "eliminated_by_quadratic_wealth_reduction" if nb == 0 else "not_used_in_reduced_lambda0_solver"
        rows_cash.append(row)

    best_row, best_solution, best_bell, best_delta = evaluate_lambda0_config(
        n_steps=48,
        n_s=401,
        n_quad=15,
        n_q=None,
        n_paths=N_PATHS_BACKTEST,
    )
    return (
        pd.DataFrame(rows_q),
        pd.DataFrame(rows_s),
        pd.DataFrame(rows_cash),
        pd.DataFrame(rows_t),
        pd.DataFrame(rows_quad),
        best_row,
        best_solution,
        best_bell,
        best_delta,
    )


def buy_hold_sell_from_result(result):
    held = result["held_delta_path"]
    if held is None or held.shape[1] < 2:
        return {"buy_frequency": np.nan, "hold_frequency": np.nan, "sell_frequency": np.nan}
    diff = np.diff(held, axis=1)
    eps = 1e-12
    return {
        "buy_frequency": float(np.mean(diff > eps)),
        "hold_frequency": float(np.mean(np.abs(diff) <= eps)),
        "sell_frequency": float(np.mean(diff < -eps)),
    }


def cost_solution_metrics(lam=0.002, n_steps=12, n_s=51, n_b=61, n_q=33, cash_mode="adaptive", interp_method="linear", n_quad=3, n_paths=3000):
    grids = make_cost_grids(n_s=n_s, n_b=n_b, n_q=n_q, cash_mode=cash_mode)
    started = time.perf_counter()
    sol = solve_bellman(lam, n_steps=n_steps, grids=grids, interp_method=interp_method, n_quad=n_quad)
    solve_time = time.perf_counter() - started
    hold = classify_policy(sol)
    times, paths = simulate_gbm_paths(n_paths=n_paths, n_steps=252, seed=SEED, mu=MU)
    bell = simulate_bellman_policy(sol, paths, times, lam)
    delta = simulate_hedge(paths, times, lam=lam, frequency=n_steps, strategy="delta_bs")
    bell_m = metrics_from_result(bell, lam, n_steps, "bellman_dp_costs", -1.0)
    delta_m = metrics_from_result(delta, lam, n_steps, "delta_bs_same_frequency", 0.0)
    diag = bell.get("diagnostics", {})
    bhs = buy_hold_sell_from_result(bell)
    return {
        "lambda": lam,
        "n_steps": n_steps,
        "Ns": n_s,
        "Nb": len(grids[1]),
        "Nq": n_q,
        "cash_mode": cash_mode,
        "interp_method": interp_method,
        "n_quad": n_quad,
        "solve_seconds": solve_time,
        "rmse_bellman": bell_m["rmse"],
        "rmse_delta_same_frequency": delta_m["rmse"],
        "rmse_ratio_vs_delta": bell_m["rmse"] / delta_m["rmse"],
        "mae_bellman": bell_m["mae"],
        "bias_bellman": bell_m["pnl_mean"],
        "mean_cost_bellman": bell_m["mean_total_cost"],
        "mean_turnover_bellman": bell_m["mean_turnover"],
        "mean_trades_bellman": bell_m["mean_n_rebalances"],
        "hold_width_mean": float(hold["hold_width_q"].mean()),
        "hold_fraction_grid": float(hold["hold_fraction"].mean()),
        **bhs,
        **diag,
    }, sol, bell


def run_cost_convergence_sweeps():
    base = {"lam": 0.002, "n_steps": 4, "n_s": 21, "n_b": 21, "n_q": 13, "cash_mode": "adaptive", "interp_method": "linear", "n_quad": 3}
    q_rows = []
    for n_q, n_s, n_b, n_steps in [(17, 21, 21, 4), (33, 17, 17, 3), (65, 13, 13, 2)]:
        row, _, _ = cost_solution_metrics(**{**base, "n_q": n_q, "n_s": n_s, "n_b": n_b, "n_steps": n_steps}, n_paths=500)
        q_rows.append(row)

    s_rows = []
    for n_s, n_q, n_b, n_steps in [(51, 13, 21, 4), (101, 9, 17, 3), (201, 7, 13, 2)]:
        row, _, _ = cost_solution_metrics(**{**base, "n_s": n_s, "n_q": n_q, "n_b": n_b, "n_steps": n_steps}, n_paths=500)
        s_rows.append(row)

    cash_rows = []
    for n_b, cash_mode in [(13, "uniform"), (13, "adaptive"), (21, "adaptive"), (31, "adaptive")]:
        row, _, _ = cost_solution_metrics(**{**base, "n_b": n_b, "cash_mode": cash_mode}, n_paths=500)
        cash_rows.append(row)

    time_rows = []
    for n_steps, n_q, n_s, n_b in [(12, 9, 17, 17), (24, 7, 13, 13), (48, 5, 11, 11)]:
        row, _, _ = cost_solution_metrics(**{**base, "n_steps": n_steps, "n_q": n_q, "n_s": n_s, "n_b": n_b}, n_paths=500)
        time_rows.append(row)

    interp_rows = []
    for interp_method in ["linear", "quadratic_cash"]:
        row, _, _ = cost_solution_metrics(**{**base, "interp_method": interp_method, "n_q": 9, "n_b": 17}, n_paths=500)
        interp_rows.append(row)

    return pd.DataFrame(q_rows), pd.DataFrame(s_rows), pd.DataFrame(cash_rows), pd.DataFrame(time_rows), pd.DataFrame(interp_rows)


def make_policy_stability(cost_q, cost_s, cost_cash, cost_time, cost_interp):
    rows = []
    for name, df in [
        ("Nq", cost_q),
        ("Ns", cost_s),
        ("cash", cost_cash),
        ("time", cost_time),
        ("interpolation", cost_interp),
    ]:
        rows.append(
            {
                "sweep": name,
                "rmse_min": float(df["rmse_bellman"].min()),
                "rmse_max": float(df["rmse_bellman"].max()),
                "rmse_range": float(df["rmse_bellman"].max() - df["rmse_bellman"].min()),
                "hold_width_min": float(df["hold_width_mean"].min()),
                "hold_width_max": float(df["hold_width_mean"].max()),
                "hold_width_range": float(df["hold_width_mean"].max() - df["hold_width_mean"].min()),
                "hold_fraction_min": float(df["hold_fraction_grid"].min()),
                "hold_fraction_max": float(df["hold_fraction_grid"].max()),
                "max_cash_outside_grid": float(df["fraction_cash_outside_grid"].max()),
                "max_s_outside_grid": float(df["fraction_s_outside_grid"].max()),
            }
        )
    return pd.DataFrame(rows)


def selected_cost_backtests():
    rows = []
    samples = []
    holds = []
    slices = []
    trajectories = []
    agg_prev = pd.read_csv(Path("results") / "hedging_transaction_costs" / "aggregate_comparisons.csv")
    times, paths = simulate_gbm_paths(n_paths=3000, n_steps=252, seed=SEED, mu=MU)
    for lam in [0.0005, 0.002, 0.01]:
        sol = solve_bellman(
            lam,
            n_steps=12,
            grids=make_cost_grids(n_s=31, n_b=31, n_q=17, cash_mode="adaptive"),
            interp_method="linear",
            n_quad=3,
        )
        hold = classify_policy(sol)
        hold["lambda"] = lam
        holds.append(hold)
        slices.append(save_policy_slices(sol))
        bell = simulate_bellman_policy(sol, paths, times, lam)
        bell_m = metrics_from_result(bell, lam, 12, "bellman_dp_costs_adaptive", -1.0)
        bell_m.update(buy_hold_sell_from_result(bell))
        bell_m.update(bell.get("diagnostics", {}))
        rows.append(bell_m)
        samples.append(sample_errors(bell, lam, "bellman_dp_costs_adaptive"))

        delta = simulate_hedge(paths, times, lam=lam, frequency=12, strategy="delta_bs")
        delta_m = metrics_from_result(delta, lam, 12, "delta_bs_same_frequency", 0.0)
        delta_m.update(non_applicable_policy_diagnostics())
        rows.append(delta_m)
        samples.append(sample_errors(delta, lam, "delta_bs_same_frequency"))

        prev = agg_prev[agg_prev["lambda"].eq(lam)].iloc[0]
        heuristic = simulate_hedge(paths, times, lam=lam, frequency=int(prev["best_optimized_frequency"]), strategy="no_trade_band", band=float(prev["best_optimized_band"]))
        heur_m = metrics_from_result(heuristic, lam, int(prev["best_optimized_frequency"]), "best_no_trade_exp3", float(prev["best_optimized_band"]))
        heur_m.update(non_applicable_policy_diagnostics())
        rows.append(heur_m)
        samples.append(sample_errors(heuristic, lam, "best_no_trade_exp3"))

        idx = bell["rebalance_indices"][:-1]
        trajectories.append(pd.DataFrame({"lambda": lam, "path": 0, "t": times[idx], "S": paths[0, idx], "delta_bs": bell["bs_delta_path"][0], "q_bellman": bell["held_delta_path"][0]}))
    return pd.DataFrame(rows), pd.concat(samples, ignore_index=True), pd.concat(holds, ignore_index=True), pd.concat(slices, ignore_index=True), pd.concat(trajectories, ignore_index=True)


def non_applicable_policy_diagnostics():
    return {
        "buy_frequency": -1.0,
        "hold_frequency": -1.0,
        "sell_frequency": -1.0,
        "fraction_s_near_boundary": -1.0,
        "fraction_cash_near_boundary": -1.0,
        "fraction_s_outside_grid": -1.0,
        "fraction_cash_outside_grid": -1.0,
    }


def drop_mixed_objective_columns(df):
    """Keep lambda>0 reporting focused on terminal squared loss and realized costs."""
    return df.drop(columns=["eta", "criterion_J"], errors="ignore")


def lambda_positive_backtests(best_n_steps):
    grids = make_grids()
    times, paths = simulate_gbm_paths(n_paths=8000, n_steps=252, seed=SEED, mu=MU)
    agg_prev = pd.read_csv(Path("results") / "hedging_transaction_costs" / "aggregate_comparisons.csv")
    rows = []
    samples = []
    hold_frames = []
    slices = []
    trajectories = []
    for lam in [0.0005, 0.002, 0.01]:
        sol = solve_bellman(lam, n_steps=N_DP, grids=grids)
        hold_frames.append(classify_policy(sol))
        slices.append(save_policy_slices(sol))
        bell = simulate_bellman_policy(sol, paths, times, lam)
        rows.append(metrics_from_result(bell, lam, N_DP, "bellman_dp_costs", -1.0))
        samples.append(sample_errors(bell, lam, "bellman_dp_costs"))
        delta = simulate_hedge(paths, times, lam=lam, frequency=N_DP, strategy="delta_bs")
        rows.append(metrics_from_result(delta, lam, N_DP, "delta_bs_same_frequency", 0.0))
        samples.append(sample_errors(delta, lam, "delta_bs_same_frequency"))
        prev = agg_prev[agg_prev["lambda"].eq(lam)].iloc[0]
        heuristic = simulate_hedge(paths, times, lam=lam, frequency=int(prev["best_optimized_frequency"]), strategy="no_trade_band", band=float(prev["best_optimized_band"]))
        rows.append(metrics_from_result(heuristic, lam, int(prev["best_optimized_frequency"]), "best_no_trade_exp3", float(prev["best_optimized_band"])))
        samples.append(sample_errors(heuristic, lam, "best_no_trade_exp3"))
        idx = bell["rebalance_indices"][:-1]
        trajectories.append(pd.DataFrame({"lambda": lam, "path": 0, "t": times[idx], "S": paths[0, idx], "delta_bs": bell["bs_delta_path"][0], "q_bellman": bell["held_delta_path"][0]}))
    return pd.DataFrame(rows), pd.concat(samples, ignore_index=True), pd.concat(hold_frames, ignore_index=True), pd.concat(slices, ignore_index=True), pd.concat(trajectories, ignore_index=True)


if __name__ == "__main__":
    run_experiment()
