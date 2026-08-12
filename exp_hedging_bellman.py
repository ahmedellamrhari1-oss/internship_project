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


def make_local_s_grid(n_s, mode="strike_dense"):
    """Stock grid with fixed tails and optional extra density near the strike."""
    if mode == "log_uniform":
        return np.exp(np.linspace(np.log(35.0), np.log(260.0), n_s))
    if mode != "strike_dense":
        raise ValueError(f"Unknown S grid mode: {mode}")
    fine = np.exp(np.linspace(np.log(35.0), np.log(260.0), 20001))
    log_distance = (np.log(fine / K) / 0.24) ** 2
    density = 1.0 + 4.0 * np.exp(-0.5 * log_distance)
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(fine))))
    grid = np.interp(np.linspace(0.0, cdf[-1], n_s), cdf, fine)
    grid[np.argmin(np.abs(grid - K))] = K
    return np.sort(grid)


def make_refined_unit_grid(n, centers=None):
    """Grid on [0,1], densified around empirical policy switching locations."""
    if not centers:
        return np.linspace(0.0, 1.0, n)
    fine = np.linspace(0.0, 1.0, 20001)
    density = np.ones_like(fine)
    for center in np.asarray(centers, dtype=float):
        density += 2.5 * np.exp(-0.5 * ((fine - center) / 0.035) ** 2)
    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(fine))))
    grid = np.interp(np.linspace(0.0, cdf[-1], n), cdf, fine)
    return np.unique(np.r_[0.0, grid[1:-1], 1.0])


def quantile_cash_grid(cash_samples, n_b, safety_fraction=0.15):
    """Empirical cash grid: quantile-spaced interior plus a tail safety margin."""
    samples = np.asarray(cash_samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    # A common initial endowment creates a large atom. It needs one grid node,
    # not many near-duplicates that waste upper-tail resolution.
    samples = np.unique(samples)
    if samples.size < 20:
        raise ValueError("At least 20 finite cash states are required")
    probs = np.linspace(0.001, 0.999, n_b)
    grid = np.quantile(samples, probs)
    span = max(float(grid[-1] - grid[0]), 1.0)
    margin = safety_fraction * span
    grid[0] = min(grid[0] - margin, float(samples.min()) - margin)
    grid[-1] = max(grid[-1] + margin, float(samples.max()) + margin)
    # Strict monotonicity is required by interpolation.
    eps = max(1e-8, span * 1e-10)
    for j in range(1, len(grid)):
        grid[j] = max(grid[j], grid[j - 1] + eps)
    return grid


def make_cost_grids(
    n_s=51,
    n_b=61,
    n_q=33,
    cash_mode="adaptive",
    cash_samples=None,
    s_mode="strike_dense",
    q_centers=None,
):
    s_grid = make_local_s_grid(n_s, mode=s_mode)
    q_grid = make_refined_unit_grid(n_q, centers=q_centers)
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
    elif cash_mode == "empirical_quantile":
        if cash_samples is None:
            raise ValueError("cash_samples are required for empirical_quantile")
        b_grid = quantile_cash_grid(cash_samples, n_b)
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
        j_mid = np.searchsorted(b_grid, b_clamped, side="left")
        j0 = np.clip(j_mid - 1, 0, len(b_grid) - 3)
        j1 = j0 + 1
        j2 = j0 + 2
        x0, x1, x2 = b_grid[j0], b_grid[j1], b_grid[j2]
        l0 = (b_clamped - x1) * (b_clamped - x2) / ((x0 - x1) * (x0 - x2))
        l1 = (b_clamped - x0) * (b_clamped - x2) / ((x1 - x0) * (x1 - x2))
        l2 = (b_clamped - x0) * (b_clamped - x1) / ((x2 - x0) * (x2 - x1))
        row0 = l0 * values[i, j0] + l1 * values[i, j1] + l2 * values[i, j2]
        row1 = l0 * values[i + 1, j0] + l1 * values[i + 1, j1] + l2 * values[i + 1, j2]
        return (1 - ws) * row0 + ws * row1
    if method != "linear":
        raise ValueError(f"Unknown interpolation method: {method}")
    return (1 - ws) * (1 - wb) * v00 + ws * (1 - wb) * v10 + (1 - ws) * wb * v01 + ws * wb * v11


def value_surface_at_q(values, q_grid, q_query):
    """Linear interpolation of V(S,b,q) in the state-q dimension."""
    q_clamped = float(np.clip(q_query, q_grid[0], q_grid[-1]))
    hi = int(np.clip(np.searchsorted(q_grid, q_clamped, side="right"), 1, len(q_grid) - 1))
    lo = hi - 1
    weight = (q_clamped - q_grid[lo]) / (q_grid[hi] - q_grid[lo])
    return (1.0 - weight) * values[lo] + weight * values[hi]


def bellman_backup_state(s, b, q, action, lam, next_values_for_action, s_grid, b_grid, dt, interp_method="linear", n_quad=5):
    b_after = b - (action - q) * s - transaction_cost(lam, s, action, q)
    b_next = b_after * math.exp(R * dt)
    expected = 0.0
    z_nodes, z_weights = gauss_hermite_normal(n_quad)
    for z, weight in zip(z_nodes, z_weights):
        s_next = s * math.exp((MU - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z)
        expected += weight * float(interp2_on_grid(next_values_for_action, s_grid, b_grid, s_next, b_next, method=interp_method))
    return expected


def solve_bellman(lam, n_steps=N_DP, grids=None, interp_method="linear", n_quad=5, action_grid=None):
    if grids is None:
        grids = make_grids()
    s_grid, b_grid, q_grid = grids
    if action_grid is None:
        action_grid = q_grid.copy()
    action_grid = np.asarray(action_grid, dtype=float)
    if np.any(np.diff(action_grid) <= 0.0) or action_grid[0] < q_grid[0] or action_grid[-1] > q_grid[-1]:
        raise ValueError("action_grid must be strictly increasing and lie inside q_grid bounds")
    dt = T / n_steps
    rf = math.exp(R * dt)
    z_nodes, z_weights = gauss_hermite_normal(n_quad)
    s_next_by_z = s_grid[:, None] * np.exp(
        (MU - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z_nodes[None, :]
    )

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
        best = np.full_like(value_next, np.inf)
        best_action = np.full_like(value_next, action_grid[0])
        q_mesh = q_grid[:, None, None]
        s_state = s_grid[None, :, None]
        b_state = b_grid[None, None, :]
        for action in action_grid:
            next_surface = value_surface_at_q(values[n + 1], q_grid, action)
            b_next = (
                b_state
                - (action - q_mesh) * s_state
                - lam * s_state * np.abs(action - q_mesh)
            ) * rf
            candidate = np.zeros_like(best)
            for iz, weight in enumerate(z_weights):
                candidate += weight * interp2_on_grid(
                    next_surface,
                    s_grid,
                    b_grid,
                    s_next_by_z[:, iz][None, :, None],
                    b_next,
                    method=interp_method,
                )
            improve = candidate < best
            best[improve] = candidate[improve]
            best_action[improve] = action
        value_now[:] = best
        policy_now[:] = best_action
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
        "action_grid": action_grid,
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


def bellman_action_objectives(solution, n, s_values, b_values, q_values, lam, actions=None):
    s_values = np.asarray(s_values)
    b_values = np.asarray(b_values)
    q_values = np.asarray(q_values)
    dt = solution["dt"]
    z_nodes, z_weights = gauss_hermite_normal(solution.get("n_quad", 5))
    interp_method = solution.get("interp_method", "linear")
    actions = solution.get("action_grid", solution["q_grid"]) if actions is None else np.asarray(actions)
    costs_by_action = []
    for action in actions:
        b_after = b_values - (action - q_values) * s_values - transaction_cost(lam, s_values, action, q_values)
        b_next = b_after * math.exp(R * dt)
        expected = np.zeros_like(s_values, dtype=float)
        for z, weight in zip(z_nodes, z_weights):
            s_next = s_values * np.exp((MU - 0.5 * SIGMA ** 2) * dt + SIGMA * math.sqrt(dt) * z)
            expected += weight * interp2_on_grid(
                value_surface_at_q(solution["values"][n + 1], solution["q_grid"], action),
                solution["s_grid"],
                solution["b_grid"],
                s_next,
                b_next,
                method=interp_method,
            )
        costs_by_action.append(expected)
    return np.vstack(costs_by_action)


def bellman_actions_continuous(solution, n, s_values, b_values, q_values, lam, control_mode="discrete"):
    actions = solution.get("action_grid", solution["q_grid"])
    stacked = bellman_action_objectives(solution, n, s_values, b_values, q_values, lam, actions=actions)
    best_idx = np.argmin(stacked, axis=0)
    selected = actions[best_idx].astype(float, copy=True)
    if control_mode == "discrete":
        return selected
    if control_mode != "local_quadratic":
        raise ValueError(f"Unknown control mode: {control_mode}")

    # A safeguarded parabolic vertex is used only at a strict, locally convex
    # interior minimum. Otherwise the discrete minimizer is retained. This is
    # an interpolation of the Bellman objective, not a BS-guided control.
    cols = np.arange(selected.size)
    flat_idx = best_idx.ravel()
    flat_cost = stacked.reshape(len(actions), -1)
    eligible = (flat_idx > 0) & (flat_idx < len(actions) - 1)
    use_cols = cols[eligible]
    ii = flat_idx[eligible]
    if use_cols.size:
        x0, x1, x2 = actions[ii - 1], actions[ii], actions[ii + 1]
        y0 = flat_cost[ii - 1, use_cols]
        y1 = flat_cost[ii, use_cols]
        y2 = flat_cost[ii + 1, use_cols]
        # Unequal-grid quadratic vertex from local divided differences.
        d01 = (y1 - y0) / (x1 - x0)
        d12 = (y2 - y1) / (x2 - x1)
        curvature = (d12 - d01) / (x2 - x0)
        vertex = 0.5 * (x0 + x1 - d01 / np.where(curvature > 0.0, curvature, 1.0))
        stable = (curvature > 1e-10) & (vertex >= x0) & (vertex <= x2)
        # Do not smooth across the transaction-cost kink a=q.
        q_flat = np.asarray(q_values).ravel()[eligible]
        stable &= ~((x0 < q_flat) & (q_flat < x2))
        out = selected.ravel()
        out[use_cols[stable]] = vertex[stable]
        selected = out.reshape(selected.shape)
    return selected


def simulate_bellman_policy(solution, paths, times, lam, control_mode="discrete", record_states=False):
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
    transition_s_outside = 0
    transition_b_outside = 0
    transition_checks_s = 0
    transition_checks_b = 0
    state_frames = []
    hold_count = 0
    buy_count = 0
    sell_count = 0

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
        if record_states:
            state_frames.append(pd.DataFrame({"time_index": n, "S": s_i, "cash": cash, "q": q}))
        actions = bellman_actions_continuous(solution, n, s_i, cash, q, lam, control_mode=control_mode)
        hold_count += int(np.sum(np.abs(actions - q) <= 1e-12))
        buy_count += int(np.sum(actions - q > 1e-12))
        sell_count += int(np.sum(actions - q < -1e-12))
        b_transition = (cash - (actions - q) * s_i - transaction_cost(lam, s_i, actions, q)) * math.exp(R * solution["dt"])
        transition_b_outside += int(np.sum((b_transition < solution["b_grid"][0]) | (b_transition > solution["b_grid"][-1])))
        transition_checks_b += n_paths
        z_nodes, _ = gauss_hermite_normal(solution.get("n_quad", 5))
        s_transition = s_i[:, None] * np.exp(
            (MU - 0.5 * SIGMA ** 2) * solution["dt"]
            + SIGMA * math.sqrt(solution["dt"]) * z_nodes[None, :]
        )
        transition_s_outside += int(np.sum((s_transition < solution["s_grid"][0]) | (s_transition > solution["s_grid"][-1])))
        transition_checks_s += s_transition.size
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
            "fraction_near_boundaries": (s_near_boundary + b_near_boundary) / (2 * n_state_checks),
            "fraction_transition_s_outside_grid": transition_s_outside / transition_checks_s,
            "fraction_transition_cash_outside_grid": transition_b_outside / transition_checks_b,
            "hold_fraction_realized": hold_count / n_state_checks,
            "buy_fraction_realized": buy_count / n_state_checks,
            "sell_fraction_realized": sell_count / n_state_checks,
        },
        "state_samples": pd.concat(state_frames, ignore_index=True) if state_frames else None,
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
        action_grid = solution.get("action_grid", solution["q_grid"])
        action_tol = 0.5 * float(np.min(np.diff(action_grid))) + 1e-12
        hold = np.isclose(diff, 0.0, atol=action_tol)
        buy = diff > 0
        sell = diff < 0
        for is_, s in enumerate(solution["s_grid"]):
            widths = []
            for ib in range(len(solution["b_grid"])):
                hold_q = solution["q_grid"][hold[:, is_, ib]]
                widths.append(float(hold_q.max() - hold_q.min()) if hold_q.size else 0.0)
            rows.append(
                {
                    "lambda": solution["lambda"],
                    "time_index": n,
                    "t": n * solution["dt"],
                    "S": s,
                    "hold_fraction": float(np.mean(hold[:, is_, :])),
                    "buy_fraction": float(np.mean(buy[:, is_, :])),
                    "sell_fraction": float(np.mean(sell[:, is_, :])),
                    "hold_width_q": float(np.mean(widths)),
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


def run_legacy_experiment():
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


def policy_boundary_centers(solution):
    """Empirical q locations at which the grid policy changes BUY/HOLD/SELL regime."""
    centers = []
    q_grid = solution["q_grid"]
    for policy in solution["policies"]:
        sign = np.sign(policy - q_grid[:, None, None])
        changes = sign[1:] != sign[:-1]
        counts = changes.sum(axis=(1, 2))
        centers.extend((0.5 * (q_grid[:-1] + q_grid[1:]))[counts > 0].tolist())
    if not centers:
        return []
    return np.quantile(np.asarray(centers), np.linspace(0.1, 0.9, 7)).tolist()


def pilot_state_audit(paths, times, lam=0.002):
    """Coarse policy used only to locate occupied state regions, never to use BS controls."""
    grids = make_cost_grids(n_s=31, n_b=41, n_q=17, cash_mode="adaptive", s_mode="strike_dense")
    solution = solve_bellman(lam, n_steps=8, grids=grids, interp_method="quadratic_cash", n_quad=3)
    result = simulate_bellman_policy(solution, paths, times, lam, record_states=True)
    states = result["state_samples"]
    rows = []
    for name in ["S", "cash", "q"]:
        values = states[name].to_numpy()
        for probability in [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]:
            rows.append({"state": name, "probability": probability, "quantile": float(np.quantile(values, probability))})
    return solution, result, states, pd.DataFrame(rows), policy_boundary_centers(solution)


def cost_solution_metrics(
    lam=0.002,
    n_steps=8,
    n_s=31,
    n_b=31,
    n_q=17,
    n_action=33,
    cash_mode="empirical_quantile",
    cash_samples=None,
    s_mode="strike_dense",
    q_mode="uniform",
    q_centers=None,
    action_mode="uniform",
    interp_method="linear",
    control_mode="discrete",
    n_quad=3,
    n_paths=2000,
    times=None,
    paths=None,
    config_id="",
):
    if cash_mode == "empirical_quantile" and cash_samples is None:
        # Backward-compatible diagnostic default; production sweeps always pass
        # pilot samples explicitly.
        cash_mode = "adaptive"
    centers = q_centers if q_mode == "boundary_refined" else None
    grids = make_cost_grids(
        n_s=n_s,
        n_b=n_b,
        n_q=n_q,
        cash_mode=cash_mode,
        cash_samples=cash_samples,
        s_mode=s_mode,
        q_centers=centers,
    )
    action_centers = q_centers if action_mode == "boundary_refined" else None
    action_grid = make_refined_unit_grid(n_action, centers=action_centers)
    started = time.perf_counter()
    sol = solve_bellman(
        lam,
        n_steps=n_steps,
        grids=grids,
        interp_method=interp_method,
        n_quad=n_quad,
        action_grid=action_grid,
    )
    solve_time = time.perf_counter() - started
    hold = classify_policy(sol)
    if paths is None or times is None:
        times, paths = simulate_gbm_paths(n_paths=n_paths, n_steps=252, seed=SEED, mu=MU)
    else:
        paths = paths[:n_paths]
    bell = simulate_bellman_policy(sol, paths, times, lam, control_mode=control_mode)
    delta = simulate_hedge(paths, times, lam=lam, frequency=n_steps, strategy="delta_bs")
    bell_m = metrics_from_result(bell, lam, n_steps, "bellman_dp_costs", -1.0)
    delta_m = metrics_from_result(delta, lam, n_steps, "delta_bs_same_frequency", 0.0)
    diag = bell.get("diagnostics", {})
    bhs = {
        "buy_frequency": diag.get("buy_fraction_realized", np.nan),
        "hold_frequency": diag.get("hold_fraction_realized", np.nan),
        "sell_frequency": diag.get("sell_fraction_realized", np.nan),
    }
    return {
        "lambda": lam,
        "n_steps": n_steps,
        "Ns": n_s,
        "Nb": len(grids[1]),
        "Nq": n_q,
        "Na": len(action_grid),
        "cash_mode": cash_mode,
        "s_mode": s_mode,
        "q_mode": q_mode,
        "action_mode": action_mode,
        "interp_method": interp_method,
        "control_mode": control_mode,
        "n_quad": n_quad,
        "config_id": config_id,
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
        "HOLD_width": float(hold["hold_width_q"].mean()),
        "HOLD_fraction": float(bell["diagnostics"]["hold_fraction_realized"]),
        "s_grid_min": float(grids[0][0]),
        "s_grid_max": float(grids[0][-1]),
        "cash_grid_min": float(grids[1][0]),
        "cash_grid_max": float(grids[1][-1]),
        "q_grid_min": float(grids[2][0]),
        "q_grid_max": float(grids[2][-1]),
        **bhs,
        **diag,
    }, sol, bell


def run_cost_convergence_sweeps(times=None, paths=None, cash_samples=None, q_centers=None, n_paths=2000):
    """True one-factor-at-a-time sweeps on common Monte Carlo paths."""
    if paths is None or times is None:
        times, paths = simulate_gbm_paths(n_paths=n_paths, n_steps=252, seed=SEED, mu=MU)
    base = {
        "lam": 0.002,
        "n_steps": 8,
        "n_s": 31,
        "n_b": 31,
        "n_q": 17,
        "n_action": 33,
        "cash_mode": "empirical_quantile" if cash_samples is not None else "adaptive",
        "cash_samples": cash_samples,
        "s_mode": "strike_dense",
        "q_mode": "uniform",
        "q_centers": q_centers,
        "action_mode": "uniform",
        "interp_method": "linear",
        "control_mode": "discrete",
        "n_quad": 3,
        "n_paths": n_paths,
        "times": times,
        "paths": paths,
    }

    def sweep(parameter, values):
        rows = []
        for value in values:
            row, _, _ = cost_solution_metrics(
                **{**base, parameter: value, "config_id": f"OFAT_{parameter}_{value}"}
            )
            row["sweep_parameter"] = parameter
            row["sweep_value"] = str(value)
            rows.append(row)
        return pd.DataFrame(rows)

    results = {
        "q": sweep("n_q", [9, 17, 25]),
        "S": sweep("n_s", [21, 31, 45]),
        "cash": sweep("n_b", [21, 31, 45]),
        "time": sweep("n_steps", [4, 8, 12]),
        "quadrature": sweep("n_quad", [3, 5, 7]),
        "interpolation": sweep("interp_method", ["linear", "quadratic_cash"]),
        "action": sweep("n_action", [17, 33, 65]),
        "S_mode": sweep("s_mode", ["log_uniform", "strike_dense"]),
        "q_mode": sweep("q_mode", ["uniform", "boundary_refined"]),
        "action_mode": sweep("action_mode", ["uniform", "boundary_refined"]),
        "control_eval": sweep("control_mode", ["discrete", "local_quadratic"]),
    }
    return results


def combined_cost_configurations():
    """Configurations chosen after the OFAT study identified time and cash resolution."""
    return [
        {"config_id": "reference_OFAT", "n_steps": 8, "n_s": 31, "n_b": 31, "n_q": 17, "n_action": 33, "n_quad": 3},
        {"config_id": "combined_mid", "n_steps": 12, "n_s": 31, "n_b": 45, "n_q": 17, "n_action": 33, "n_quad": 5},
        {"config_id": "time_cash_fine", "n_steps": 16, "n_s": 31, "n_b": 61, "n_q": 17, "n_action": 33, "n_quad": 7},
        {"config_id": "all_dimensions_fine", "n_steps": 16, "n_s": 45, "n_b": 61, "n_q": 25, "n_action": 65, "n_quad": 7},
    ]


def evaluate_combined_configurations(times, paths, cash_samples, q_centers, n_paths=3000):
    rows = []
    for config in combined_cost_configurations():
        row, _, _ = cost_solution_metrics(
            lam=0.002,
            cash_mode="empirical_quantile",
            cash_samples=cash_samples,
            s_mode="strike_dense",
            q_mode="uniform",
            q_centers=q_centers,
            action_mode="uniform",
            interp_method="linear",
            control_mode="discrete",
            times=times,
            paths=paths,
            n_paths=n_paths,
            **config,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_final_cost_validation(best_config, cash_samples, q_centers, n_paths=10000):
    """Common-path comparison required for the selected lambda=0.002 policy."""
    times, paths = simulate_gbm_paths(n_paths=n_paths, n_steps=252, seed=SEED, mu=MU)
    row, solution, bell = cost_solution_metrics(
        lam=0.002,
        cash_mode="empirical_quantile",
        cash_samples=cash_samples,
        s_mode="strike_dense",
        q_mode="uniform",
        q_centers=q_centers,
        action_mode="uniform",
        interp_method="linear",
        control_mode="discrete",
        times=times,
        paths=paths,
        n_paths=n_paths,
        **best_config,
    )
    frequency = int(best_config["n_steps"])
    delta = simulate_hedge(paths, times, lam=0.002, frequency=frequency, strategy="delta_bs")
    aggregate = pd.read_csv(Path("results") / "hedging_transaction_costs" / "aggregate_comparisons.csv")
    previous = aggregate[aggregate["lambda"].eq(0.002)].iloc[0]
    heuristic = simulate_hedge(
        paths,
        times,
        lam=0.002,
        frequency=int(previous["best_optimized_frequency"]),
        strategy="no_trade_band",
        band=float(previous["best_optimized_band"]),
    )

    bell_metrics = metrics_from_result(bell, 0.002, frequency, "bellman_costs_selected", -1.0)
    bell_metrics.update(bell["diagnostics"])
    bell_metrics.update({"solve_seconds": row["solve_seconds"], "HOLD_fraction": row["HOLD_fraction"], "HOLD_width": row["HOLD_width"]})
    delta_metrics = metrics_from_result(delta, 0.002, frequency, "delta_bs_same_frequency", 0.0)
    heuristic_metrics = metrics_from_result(
        heuristic,
        0.002,
        int(previous["best_optimized_frequency"]),
        "best_no_trade_exp3A",
        float(previous["best_optimized_band"]),
    )
    metrics = pd.DataFrame([bell_metrics, delta_metrics, heuristic_metrics])
    metrics["bias"] = metrics["pnl_mean"]
    metrics["mean_transaction_cost"] = metrics["mean_total_cost"]
    metrics["turnover"] = metrics["mean_turnover"]
    metrics["number_of_trades"] = metrics["mean_n_rebalances"]
    errors = pd.concat(
        [
            sample_errors(bell, 0.002, "bellman_costs_selected", n=n_paths),
            sample_errors(delta, 0.002, "delta_bs_same_frequency", n=n_paths),
            sample_errors(heuristic, 0.002, "best_no_trade_exp3A", n=n_paths),
        ],
        ignore_index=True,
    )
    return metrics, errors, solution, bell, row


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


def plot_cost_refinement_results(sweeps, combined, final_metrics, cash_samples, selected_solution):
    for name, df in sweeps.items():
        x = np.arange(len(df))
        labels = df["sweep_value"].astype(str).tolist()
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(x, df["rmse_bellman"], marker="o", label="Bellman RMSE")
        ax.set_xticks(x, labels)
        ax.set_xlabel(df["sweep_parameter"].iloc[0])
        ax.set_ylabel("terminal hedging RMSE")
        ax.grid(True, alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(x, df["solve_seconds"], marker="s", color="tab:orange", label="solve time")
        ax2.set_ylabel("solve seconds")
        lines, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines_2, labels_1 + labels_2, loc="best")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"ofat_{name}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(combined["solve_seconds"], combined["rmse_bellman"], s=55)
    for row in combined.itertuples(index=False):
        ax.annotate(row.config_id, (row.solve_seconds, row.rmse_bellman), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("solve seconds")
    ax.set_ylabel("Bellman RMSE")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "combined_accuracy_vs_cost.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(final_metrics["strategy"], final_metrics["rmse"])
    ax.set_ylabel("terminal hedging RMSE")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "final_common_path_rmse.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cash_samples, bins=100, density=True, alpha=0.55, label="pilot cash states")
    grid = selected_solution["b_grid"]
    ax.vlines(grid, 0.0, ax.get_ylim()[1] * 0.12, color="tab:red", alpha=0.25, linewidth=0.7, label="cash grid")
    ax.set_xlabel("cash state b")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cash_state_occupancy_and_grid.png", dpi=150)
    plt.close(fig)


def run_experiment():
    """Reproducible lambda>0 refinement study; the lambda=0 branch is untouched."""
    ensure_dirs()
    started = time.perf_counter()
    sweep_times, sweep_paths = simulate_gbm_paths(n_paths=3000, n_steps=252, seed=SEED, mu=MU)
    _, pilot, states, quantiles, q_centers = pilot_state_audit(sweep_paths[:2000], sweep_times)
    states.to_csv(RESULT_DIR / "state_occupancy_pilot.csv", index=False)
    quantiles.to_csv(RESULT_DIR / "state_occupancy_quantiles.csv", index=False)
    write_json(
        RESULT_DIR / "pilot_metadata.json",
        {"seed": SEED, "n_paths": 2000, "boundary_centers": q_centers, "pilot_diagnostics": pilot["diagnostics"]},
    )

    sweeps = run_cost_convergence_sweeps(
        sweep_times,
        sweep_paths[:2000],
        states["cash"].to_numpy(),
        q_centers,
        n_paths=2000,
    )
    output_names = {
        "q": "convergence_costs_q.csv",
        "S": "convergence_costs_S.csv",
        "cash": "convergence_costs_cash.csv",
        "time": "convergence_costs_time.csv",
        "quadrature": "convergence_costs_quadrature.csv",
        "interpolation": "interpolation_cost_diagnostics.csv",
        "action": "convergence_costs_action.csv",
        "S_mode": "convergence_costs_S_mode.csv",
        "q_mode": "convergence_costs_q_mode.csv",
        "action_mode": "convergence_costs_action_mode.csv",
        "control_eval": "convergence_costs_control_eval.csv",
    }
    for name, frame in sweeps.items():
        frame.to_csv(RESULT_DIR / output_names[name], index=False)

    combined = evaluate_combined_configurations(
        sweep_times,
        sweep_paths,
        states["cash"].to_numpy(),
        q_centers,
        n_paths=3000,
    )
    combined.to_csv(RESULT_DIR / "combined_configurations.csv", index=False)
    eligible = combined[~combined["config_id"].eq("reference_OFAT")]
    best_id = str(eligible.sort_values(["rmse_bellman", "solve_seconds"]).iloc[0]["config_id"])
    best_config = next(config for config in combined_cost_configurations() if config["config_id"] == best_id)
    write_json(RESULT_DIR / "best_cost_config.json", best_config)

    final_metrics, final_errors, solution, bell, final_row = run_final_cost_validation(
        best_config,
        states["cash"].to_numpy(),
        q_centers,
        n_paths=N_PATHS_BACKTEST,
    )
    final_metrics.to_csv(RESULT_DIR / "final_validation_metrics.csv", index=False)
    final_errors.to_csv(RESULT_DIR / "final_validation_terminal_errors.csv", index=False)
    classify_policy(solution).to_csv(RESULT_DIR / "best_policy_hold_regions.csv", index=False)
    save_policy_slices(solution).to_csv(RESULT_DIR / "best_policy_slices.csv", index=False)
    combined[[
        "config_id", "rmse_bellman", "solve_seconds", "HOLD_fraction", "HOLD_width",
        "fraction_near_boundaries", "fraction_s_outside_grid", "fraction_cash_outside_grid",
        "fraction_transition_s_outside_grid", "fraction_transition_cash_outside_grid",
    ]].to_csv(RESULT_DIR / "policy_stability.csv", index=False)
    plot_cost_refinement_results(sweeps, combined, final_metrics, states["cash"].to_numpy(), solution)

    bell_final = final_metrics[final_metrics["strategy"].eq("bellman_costs_selected")].iloc[0]
    delta_final = final_metrics[final_metrics["strategy"].eq("delta_bs_same_frequency")].iloc[0]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.perf_counter() - started,
        "lambda": 0.002,
        "seed": SEED,
        "sweep_paths": 2000,
        "selection_paths": 3000,
        "final_paths": N_PATHS_BACKTEST,
        "financial_formulation_unchanged": {"state": ["S", "b", "q"], "cost": "lambda*S*abs(a-q)", "objective": "E[(W_T-H)^2]"},
        "black_scholes_use": "benchmark and financially justified initial endowment only; never used by solve_bellman or action selection",
        "selected_config": best_config,
        "selected_final_diagnostics": final_row,
        "bellman_rmse": float(bell_final["rmse"]),
        "delta_same_frequency_rmse": float(delta_final["rmse"]),
        "bellman_to_delta_rmse_ratio": float(bell_final["rmse"] / delta_final["rmse"]),
        "ofat_bottleneck": "time and cash resolution; state-q, stock, action and quadrature largely plateau at the tested base",
        "continuous_control_decision": "rejected: negligible RMSE change with a large HOLD-frequency change",
        "quadratic_cash_decision": "rejected: unstable on non-uniform empirical cash grids",
        "combined_configurations": combined.to_dict(orient="records"),
        "final_metrics": final_metrics.to_dict(orient="records"),
    }
    write_json(RESULT_DIR / "summary_cost_refinement.json", summary)
    write_json(
        RESULT_DIR / "config_cost_refinement.json",
        {"seed": SEED, "common_paths": True, "sweep_base": {"N": 8, "Ns": 31, "Nb": 31, "Nq": 17, "Na": 33, "n_quad": 3}},
    )
    print(json.dumps(summary, indent=2))
    return solution, final_metrics, summary


if __name__ == "__main__":
    run_experiment()
