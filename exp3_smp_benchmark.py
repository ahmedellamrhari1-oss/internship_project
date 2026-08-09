import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from merton_closed_form import MertonModel  # noqa: E402
from smp_fbsde import pathwise_diagnostics, run_smp_fbsde  # noqa: E402


RESULT_DIR = Path("results") / "exp3"
FIG_DIR = RESULT_DIR / "figures"


def archive_if_exists(path: Path):
    if path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(str(path), str(path.with_name(f"{path.stem}_{stamp}{path.suffix}.bak")))


def write_json(path: Path, data):
    archive_if_exists(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame):
    archive_if_exists(path)
    df.to_csv(path, index=False)


def run_one(model, N, M, seed, x0=1.0):
    started = time.perf_counter()
    eval_times = [0.0, 0.25 * model.T, 0.5 * model.T, 0.75 * model.T, (N - 1) / N * model.T]
    result = run_smp_fbsde(
        model,
        N=N,
        M=M,
        x0=x0,
        seed=seed,
        scheme="exact",
        eval_times=eval_times,
        return_pathwise=True,
    )
    runtime = time.perf_counter() - started
    rows = pathwise_diagnostics(model, result)
    for row in rows:
        row["runtime"] = runtime
    return rows


def run_sweep(model, Ns, Ms, seeds, fixed_N, fixed_M, x0):
    rows_N = []
    for N in Ns:
        rows_N.extend(run_one(model, N=N, M=fixed_M, seed=seeds[0], x0=x0))

    rows_M = []
    for M in Ms:
        rows_M.extend(run_one(model, N=fixed_N, M=M, seed=seeds[0], x0=x0))

    rows_seed = []
    for seed in seeds:
        rows_seed.extend(run_one(model, N=fixed_N, M=fixed_M, seed=seed, x0=x0))
    return pd.DataFrame(rows_N), pd.DataFrame(rows_M), pd.DataFrame(rows_seed)


def seed_aggregate(seed_df):
    metric_cols = [
        "RMSE_p",
        "relative_RMSE_p",
        "RMSE_q",
        "relative_RMSE_q",
        "RMSE_Hpi",
        "RMSE_pi_reconstructed",
        "relative_RMSE_pi_reconstructed",
    ]
    grouped = seed_df.groupby(["t_index", "t", "N", "M"], dropna=False)[metric_cols]
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std(ddof=1).add_suffix("_std")
    return pd.concat([mean_df, std_df], axis=1).reset_index()


def finite_no_terminal(df):
    return df[np.isfinite(df["RMSE_q"])].copy()


def plot_results(diag_df, conv_N, conv_M):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIG_DIR.glob("*.png"):
        archive_if_exists(path)

    mid_N = finite_no_terminal(conv_N[np.isclose(conv_N["t"], 0.5)])
    plt.figure()
    plt.plot(mid_N["N"], mid_N["relative_RMSE_p"], marker="o", label="p")
    plt.plot(mid_N["N"], mid_N["relative_RMSE_q"], marker="o", label="q")
    plt.plot(mid_N["N"], mid_N["RMSE_Hpi"], marker="o", label="H_pi")
    plt.xlabel("N")
    plt.ylabel("metric at t=0.5T")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(FIG_DIR / "metrics_vs_N_t05.png", dpi=160, bbox_inches="tight")
    plt.close()

    mid_M = finite_no_terminal(conv_M[np.isclose(conv_M["t"], 0.5)])
    plt.figure()
    plt.plot(mid_M["M"], mid_M["relative_RMSE_p"], marker="o", label="p")
    plt.plot(mid_M["M"], mid_M["relative_RMSE_q"], marker="o", label="q")
    plt.plot(mid_M["M"], mid_M["RMSE_Hpi"], marker="o", label="H_pi")
    plt.xlabel("M")
    plt.ylabel("metric at t=0.5T")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(FIG_DIR / "metrics_vs_M_t05.png", dpi=160, bbox_inches="tight")
    plt.close()

    d = finite_no_terminal(diag_df)
    plt.figure()
    plt.plot(d["t"], d["relative_RMSE_p"], marker="o", label="p")
    plt.plot(d["t"], d["relative_RMSE_q"], marker="o", label="q")
    plt.plot(d["t"], d["RMSE_Hpi"], marker="o", label="H_pi")
    plt.xlabel("t")
    plt.ylabel("metric")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(FIG_DIR / "diagnostics_over_time.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(d["t"], d["relative_RMSE_pi_reconstructed"], marker="o")
    plt.xlabel("t")
    plt.ylabel("relative RMSE pi_reconstructed")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.savefig(FIG_DIR / "pi_reconstructed_over_time.png", dpi=160, bbox_inches="tight")
    plt.close()


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    x0 = 1.0
    Ns = [10, 20, 50, 100, 200]
    Ms = [2000, 5000, 10000, 20000, 40000, 80000]
    seeds = [0, 1, 2, 3, 4]
    fixed_N = 50
    fixed_M = 40000

    config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "financial_parameters": {
            "r": model.r,
            "mu": model.mu,
            "sigma": model.sigma,
            "gamma": model.gamma,
            "T": model.T,
            "x0": x0,
            "alpha": model.pi_star,
        },
        "numerical_parameters": {
            "scheme": "exact_gbm",
            "N_sweep": Ns,
            "M_sweep": Ms,
            "fixed_N": fixed_N,
            "fixed_M": fixed_M,
            "seeds": seeds,
            "eval_times": ["0", "0.25T", "0.5T", "0.75T", "t_{N-1}"],
        },
    }

    conv_N, conv_M, seed_df = run_sweep(model, Ns, Ms, seeds, fixed_N, fixed_M, x0)
    diagnostics_time = seed_df[seed_df["seed"].eq(seeds[0])].copy()
    seed_summary = seed_aggregate(seed_df)

    write_json(RESULT_DIR / "config.json", config)
    write_csv(RESULT_DIR / "convergence_N.csv", conv_N)
    write_csv(RESULT_DIR / "convergence_M.csv", conv_M)
    write_csv(RESULT_DIR / "diagnostics_time.csv", diagnostics_time)
    write_csv(RESULT_DIR / "seed_replicates.csv", seed_df)
    write_csv(RESULT_DIR / "seed_summary.csv", seed_summary)
    plot_results(diagnostics_time, conv_N, conv_M)

    finite_diag = finite_no_terminal(diagnostics_time)
    mid = finite_diag.iloc[(finite_diag["t"] - 0.5).abs().argsort()[:1]]
    summary = {
        "diagnostics_seed": seeds[0],
        "fixed_N": fixed_N,
        "fixed_M": fixed_M,
        "mid_time": float(mid["t"].iloc[0]),
        "mid_relative_RMSE_p": float(mid["relative_RMSE_p"].iloc[0]),
        "mid_relative_RMSE_q": float(mid["relative_RMSE_q"].iloc[0]),
        "mid_RMSE_Hpi": float(mid["RMSE_Hpi"].iloc[0]),
        "mid_relative_RMSE_pi_reconstructed": float(mid["relative_RMSE_pi_reconstructed"].iloc[0]),
        "seed_count": len(seeds),
        "n_figures": len(list(FIG_DIR.glob("*.png"))),
    }
    write_json(RESULT_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
