import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.special import ndtr
except ImportError:
    ndtr = None


RESULT_DIR = Path("results") / "hedging_transaction_costs"
FIG_DIR = RESULT_DIR / "figures"

S0 = 100.0
K = 100.0
T = 1.0
R = 0.02
MU = 0.08
SIGMA = 0.20
N_FINE = 252
N_PATHS = 20000
SEED = 12345
LAMBDAS = [0.0, 0.0005, 0.002, 0.01]
FREQUENCIES = [12, 26, 52, 126, 252]
BAND_GRID = [0.0, 0.02, 0.05, 0.10, 0.20]
ETA = 1.0


def ensure_dirs():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    if ndtr is not None:
        return ndtr(x)
    erf = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf(x / math.sqrt(2.0)))


def call_payoff(s, k=K):
    return np.maximum(np.asarray(s, dtype=float) - k, 0.0)


def bs_call_price(s, tau, k=K, r=R, sigma=SIGMA):
    s = np.asarray(s, dtype=float)
    tau = np.asarray(tau, dtype=float)
    intrinsic = call_payoff(s, k)
    tau_safe = np.maximum(tau, 1e-14)
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * tau_safe) / (sigma * np.sqrt(tau_safe))
    d2 = d1 - sigma * np.sqrt(tau_safe)
    price = s * norm_cdf(d1) - k * np.exp(-r * tau_safe) * norm_cdf(d2)
    return np.where(tau > 0.0, price, intrinsic)


def bs_delta(s, tau, k=K, r=R, sigma=SIGMA):
    s = np.asarray(s, dtype=float)
    tau = np.asarray(tau, dtype=float)
    tau_safe = np.maximum(tau, 1e-14)
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * tau_safe) / (sigma * np.sqrt(tau_safe))
    delta = norm_cdf(d1)
    terminal_delta = np.where(s > k, 1.0, np.where(s < k, 0.0, 0.5))
    return np.where(tau > 0.0, delta, terminal_delta)


def transaction_cost(lam, s, delta_new, delta_old):
    return lam * np.asarray(s, dtype=float) * np.abs(np.asarray(delta_new) - np.asarray(delta_old))


def simulate_gbm_paths(n_paths=N_PATHS, n_steps=N_FINE, s0=S0, mu=MU, sigma=SIGMA, t=T, seed=SEED):
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    log_paths = np.cumsum(log_increments, axis=1)
    s = np.empty((n_paths, n_steps + 1), dtype=float)
    s[:, 0] = s0
    s[:, 1:] = s0 * np.exp(log_paths)
    times = np.linspace(0.0, t, n_steps + 1)
    return times, s


def rebalance_indices(frequency, n_steps=N_FINE):
    raw = np.rint(np.linspace(0, n_steps, frequency + 1)).astype(int)
    return np.unique(raw)


def target_delta(delta_bs, current_delta, strategy, band=0.0):
    if strategy == "delta_bs":
        return delta_bs
    if strategy == "no_trade_band":
        should_trade = np.abs(delta_bs - current_delta) > band
        return np.where(should_trade, delta_bs, current_delta)
    raise ValueError(f"Unknown strategy: {strategy}")


def simulate_hedge(paths, times, lam, frequency, strategy="delta_bs", band=0.0, store_paths=False):
    n_paths = paths.shape[0]
    idx = rebalance_indices(frequency, n_steps=paths.shape[1] - 1)
    option_price = float(bs_call_price(S0, T))

    cash = np.full(n_paths, option_price, dtype=float)
    delta = np.zeros(n_paths, dtype=float)
    stock_position = np.zeros(n_paths, dtype=float)
    cumulative_cost = np.zeros(n_paths, dtype=float)
    turnover = np.zeros(n_paths, dtype=float)
    n_trades = np.zeros(n_paths, dtype=float)
    held_delta_path = np.full((n_paths, len(idx) - 1), np.nan, dtype=float) if store_paths else None
    bs_delta_path = np.full((n_paths, len(idx) - 1), np.nan, dtype=float) if store_paths else None

    last_i = 0
    for j, i in enumerate(idx[:-1]):
        if i > last_i:
            cash *= np.exp(R * (times[i] - times[last_i]))

        s_i = paths[:, i]
        tau_i = T - times[i]
        delta_bs = bs_delta(s_i, tau_i)
        new_delta = target_delta(delta_bs, delta, strategy=strategy, band=band)
        d_delta = new_delta - delta
        costs = transaction_cost(lam, s_i, new_delta, delta)

        cash -= d_delta * s_i + costs
        stock_position = new_delta * s_i
        cumulative_cost += costs
        turnover += np.abs(d_delta)
        n_trades += (np.abs(d_delta) > 1e-12)
        delta = new_delta
        if store_paths:
            held_delta_path[:, j] = delta
            bs_delta_path[:, j] = delta_bs
        last_i = i

    cash *= np.exp(R * (T - times[last_i]))
    terminal_wealth = cash + delta * paths[:, -1]
    payoff = call_payoff(paths[:, -1])
    error = terminal_wealth - payoff
    pnl = error

    return {
        "error": error,
        "pnl": pnl,
        "terminal_wealth": terminal_wealth,
        "payoff": payoff,
        "cumulative_cost": cumulative_cost,
        "turnover": turnover,
        "n_trades": n_trades,
        "rebalance_indices": idx,
        "held_delta_path": held_delta_path,
        "bs_delta_path": bs_delta_path,
    }


def metrics_from_result(result, lam, frequency, strategy, band, eta=ETA):
    error = result["error"]
    pnl = result["pnl"]
    mean_cost = float(np.mean(result["cumulative_cost"]))
    mse = float(np.mean(error ** 2))
    return {
        "lambda": lam,
        "frequency": frequency,
        "strategy": strategy,
        "band": band,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(error))),
        "pnl_mean": float(np.mean(pnl)),
        "pnl_q01": float(np.quantile(pnl, 0.01)),
        "pnl_q05": float(np.quantile(pnl, 0.05)),
        "pnl_q50": float(np.quantile(pnl, 0.50)),
        "pnl_q95": float(np.quantile(pnl, 0.95)),
        "pnl_q99": float(np.quantile(pnl, 0.99)),
        "mean_total_cost": mean_cost,
        "mean_turnover": float(np.mean(result["turnover"])),
        "mean_n_rebalances": float(np.mean(result["n_trades"])),
        "mse": mse,
        "eta": eta,
        "criterion_J": mse + eta * mean_cost,
    }


def run_experiment():
    ensure_dirs()
    for path in FIG_DIR.glob("*.png"):
        path.unlink()

    started = time.perf_counter()
    times, paths = simulate_gbm_paths()
    path_sample = pd.DataFrame({"time": times, "S_path_0": paths[0]})
    path_sample.to_csv(RESULT_DIR / "sample_path.csv", index=False)

    rows = []
    error_samples = []
    best_band_rows = []
    path_rows = []

    for lam in LAMBDAS:
        for frequency in FREQUENCIES:
            delta_result = simulate_hedge(paths, times, lam=lam, frequency=frequency, strategy="delta_bs")
            rows.append(metrics_from_result(delta_result, lam, frequency, "delta_bs", 0.0))
            if frequency in (12, 52, 252):
                error_samples.append(sample_errors(delta_result, lam, frequency, "delta_bs", 0.0))

            band_results = []
            for band in BAND_GRID:
                result = simulate_hedge(paths, times, lam=lam, frequency=frequency, strategy="no_trade_band", band=band)
                metric = metrics_from_result(result, lam, frequency, "no_trade_band", band)
                rows.append(metric)
                band_results.append((metric, result))
            best_metric, best_result = min(band_results, key=lambda item: item[0]["criterion_J"])
            best_band_rows.append({**best_metric, "strategy": "optimized_no_trade_band"})
            if frequency == 252:
                error_samples.append(sample_errors(best_result, lam, frequency, "optimized_no_trade_band", best_metric["band"]))

        example_delta = simulate_hedge(paths[:1], times, lam=lam, frequency=252, strategy="delta_bs", store_paths=True)
        best_band = pd.DataFrame(best_band_rows)
        best_daily = best_band[(best_band["lambda"].eq(lam)) & (best_band["frequency"].eq(252))].iloc[0]
        example_opt = simulate_hedge(
            paths[:1],
            times,
            lam=lam,
            frequency=252,
            strategy="no_trade_band",
            band=float(best_daily["band"]),
            store_paths=True,
        )
        idx = example_delta["rebalance_indices"][:-1]
        path_rows.append(
            pd.DataFrame(
                {
                    "lambda": lam,
                    "time": times[idx],
                    "S": paths[0, idx],
                    "delta_bs": example_delta["bs_delta_path"][0],
                    "position_delta_bs": example_delta["held_delta_path"][0],
                    "position_optimized": example_opt["held_delta_path"][0],
                    "optimized_band": float(best_daily["band"]),
                }
            )
        )

    metrics = pd.concat([pd.DataFrame(rows), pd.DataFrame(best_band_rows)], ignore_index=True)
    metrics = metrics.sort_values(["lambda", "frequency", "strategy", "band"]).reset_index(drop=True)
    errors = pd.concat(error_samples, ignore_index=True)
    example_paths = pd.concat(path_rows, ignore_index=True)

    metrics.to_csv(RESULT_DIR / "metrics.csv", index=False)
    errors.to_csv(RESULT_DIR / "terminal_error_samples.csv", index=False)
    example_paths.to_csv(RESULT_DIR / "example_hedge_paths.csv", index=False)
    aggregate = make_aggregate(metrics)
    aggregate.to_csv(RESULT_DIR / "aggregate_comparisons.csv", index=False)
    plot_results(metrics, errors, example_paths)

    summary = make_summary(metrics, runtime=time.perf_counter() - started)
    write_json(RESULT_DIR / "summary.json", summary)
    write_json(RESULT_DIR / "config.json", config_dict())
    print(json.dumps(summary, indent=2))
    return metrics, summary


def sample_errors(result, lam, frequency, strategy, band, n=5000):
    n = min(n, result["error"].shape[0])
    return pd.DataFrame(
        {
            "lambda": lam,
            "frequency": frequency,
            "strategy": strategy,
            "band": band,
            "error": result["error"][:n],
            "pnl": result["pnl"][:n],
            "cost": result["cumulative_cost"][:n],
        }
    )


def make_aggregate(metrics):
    rows = []
    for lam in LAMBDAS:
        subset = metrics[metrics["lambda"].eq(lam)]
        best_delta = subset[subset["strategy"].eq("delta_bs")].sort_values("criterion_J").iloc[0]
        best_opt = subset[subset["strategy"].eq("optimized_no_trade_band")].sort_values("criterion_J").iloc[0]
        rows.append(
            {
                "lambda": lam,
                "best_delta_frequency": int(best_delta["frequency"]),
                "best_delta_J": float(best_delta["criterion_J"]),
                "best_delta_rmse": float(best_delta["rmse"]),
                "best_delta_cost": float(best_delta["mean_total_cost"]),
                "best_optimized_frequency": int(best_opt["frequency"]),
                "best_optimized_band": float(best_opt["band"]),
                "best_optimized_J": float(best_opt["criterion_J"]),
                "best_optimized_rmse": float(best_opt["rmse"]),
                "best_optimized_cost": float(best_opt["mean_total_cost"]),
                "optimized_improves_J": bool(best_opt["criterion_J"] < best_delta["criterion_J"]),
                "optimized_trades_less": bool(best_opt["mean_n_rebalances"] < best_delta["mean_n_rebalances"]),
            }
        )
    return pd.DataFrame(rows)


def make_summary(metrics, runtime):
    delta_l0 = metrics[(metrics["lambda"].eq(0.0)) & (metrics["strategy"].eq("delta_bs"))].sort_values("frequency")
    rmse_by_frequency_l0 = {
        str(int(row.frequency)): float(row.rmse)
        for row in delta_l0.itertuples(index=False)
    }
    aggregate = make_aggregate(metrics)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": runtime,
        "results_dir": str(RESULT_DIR),
        "n_paths": N_PATHS,
        "n_fine_steps": N_FINE,
        "rmse_delta_lambda_0_by_frequency": rmse_by_frequency_l0,
        "lambda_0_rmse_decreases_with_frequency": bool(np.all(np.diff(delta_l0["rmse"].to_numpy()) < 0.0)),
        "best_by_lambda": aggregate.to_dict(orient="records"),
        "n_figures": len(list(FIG_DIR.glob("*.png"))),
    }


def config_dict():
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "scipy_ndtr_available": ndtr is not None,
        "model": {
            "S0": S0,
            "K": K,
            "T": T,
            "r": R,
            "mu": MU,
            "sigma": SIGMA,
            "option": "European call",
            "initial_option_price": float(bs_call_price(S0, T)),
        },
        "simulation": {
            "scheme": "exact_gbm",
            "n_paths": N_PATHS,
            "n_fine_steps": N_FINE,
            "seed": SEED,
        },
        "experiment": {
            "lambdas": LAMBDAS,
            "frequencies_per_year": FREQUENCIES,
            "optimized_strategy": "no_trade_band_grid_search",
            "band_grid": BAND_GRID,
            "criterion": "E[(X_T-H)^2] + eta E[cumulative transaction costs]",
            "eta": ETA,
        },
    }


def plot_results(metrics, errors, example_paths):
    delta = metrics[metrics["strategy"].eq("delta_bs")]
    opt = metrics[metrics["strategy"].eq("optimized_no_trade_band")]

    for lam in LAMBDAS:
        subset = errors[errors["lambda"].eq(lam)]
        plt.figure(figsize=(10, 5))
        for (strategy, frequency), df in subset.groupby(["strategy", "frequency"]):
            if strategy == "delta_bs" and frequency not in (12, 252):
                continue
            plt.hist(df["error"], bins=80, alpha=0.45, density=True, label=f"{strategy}, N={frequency}")
        plt.xlabel("Terminal hedging error X_T - H")
        plt.ylabel("Density")
        plt.title(f"Terminal error distribution, lambda={lam:g}")
        plt.grid(True)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"terminal_errors_lambda_{lam:g}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(9, 5))
    for lam, df in delta.groupby("lambda"):
        plt.plot(df["frequency"], df["rmse"], marker="o", label=f"lambda={lam:g}")
    plt.xlabel("Rebalances per year")
    plt.ylabel("RMSE")
    plt.xscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rmse_vs_frequency.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for lam, df in delta.groupby("lambda"):
        plt.plot(df["frequency"], df["mean_total_cost"], marker="o", label=f"lambda={lam:g}")
    plt.xlabel("Rebalances per year")
    plt.ylabel("Mean transaction cost")
    plt.xscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "transaction_cost_vs_frequency.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for lam, df in delta.groupby("lambda"):
        plt.plot(df["frequency"], df["criterion_J"], marker="o", label=f"delta, lambda={lam:g}")
    for lam, df in opt.groupby("lambda"):
        plt.plot(df["frequency"], df["criterion_J"], marker="s", linestyle="--", label=f"opt, lambda={lam:g}")
    plt.xlabel("Rebalances per year")
    plt.ylabel("Criterion J")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "criterion_J_vs_frequency.png", dpi=160)
    plt.close()

    for lam, df in example_paths.groupby("lambda"):
        plt.figure(figsize=(11, 6))
        ax1 = plt.gca()
        ax1.plot(df["time"], df["S"], color="black", label="S_t")
        ax1.set_xlabel("t")
        ax1.set_ylabel("S_t")
        ax2 = ax1.twinx()
        ax2.plot(df["time"], df["delta_bs"], label="BS delta", alpha=0.8)
        ax2.plot(df["time"], df["position_optimized"], label="held optimized", alpha=0.8)
        ax2.set_ylabel("Delta / held position")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="best")
        plt.title(f"Example path and hedges, lambda={lam:g}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"example_path_delta_lambda_{lam:g}.png", dpi=160)
        plt.close()

    plt.figure(figsize=(9, 5))
    daily_delta = delta[delta["frequency"].eq(252)]
    daily_opt = opt[opt["frequency"].eq(252)]
    plt.plot(daily_delta["lambda"], daily_delta["mean_turnover"], marker="o", label="daily delta")
    plt.plot(daily_opt["lambda"], daily_opt["mean_turnover"], marker="s", label="daily optimized")
    plt.xlabel("lambda")
    plt.ylabel("Mean turnover")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "turnover_vs_lambda.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    run_experiment()
