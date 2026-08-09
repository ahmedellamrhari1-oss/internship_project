import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hjb_fdm import derivatives_from_log_grid, evaluate_solution_at_x0, pi_from_grid, solve_hjb_fdm  # noqa: E402
from merton_closed_form import MertonModel  # noqa: E402


RESULT_DIR = Path("results") / "hjb_merton_1d"
FIG_DIR = RESULT_DIR / "figures"
GAMMAS = [1.5, 3.0, 5.0]
NX_LIST = [50, 100, 200, 400]
NT_LIST = [50, 100, 200, 400]


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame):
    df.to_csv(path, index=False)


def model_for_gamma(gamma):
    return MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=gamma, T=1.0)


def interior_mask(y, trim_fraction=0.10):
    n = len(y)
    lo = max(1, int(np.floor(trim_fraction * n)))
    hi = min(n - 1, int(np.ceil((1.0 - trim_fraction) * n)))
    mask = np.zeros(n, dtype=bool)
    mask[lo:hi] = True
    return mask


def run_one(model, Nx, Nt, x0=1.0, y_min=-3.0, y_max=3.0, eps=1e-10):
    started = time.perf_counter()
    y, w, stats = solve_hjb_fdm(
        model,
        Ny=Nx,
        Nt=Nt,
        y_min=y_min,
        y_max=y_max,
        return_diagnostics=True,
        eps=eps,
    )
    runtime = time.perf_counter() - started

    x = np.exp(y)
    V_exact_grid = model.V(0.0, x)
    pi_exact_grid = model.pi_amount(x)
    pi_num, _, Vxx, pi_stats = pi_from_grid(model, y, w, eps=eps, return_diagnostics=True)
    ev = evaluate_solution_at_x0(model, y, w, x0=x0)

    mask = interior_mask(y)
    control_error = pi_num[mask] - pi_exact_grid[mask]
    V_exact_x0 = model.V(0.0, x0)
    pi_exact_x0 = model.pi_amount(x0)

    row = {
        "gamma": model.gamma,
        "Nx": Nx,
        "Nt": Nt,
        "dy": stats["dy"],
        "dt": stats["dt"],
        "lambda": stats["lambda"],
        "x0": x0,
        "y0": ev["y0"],
        "nearest_y": ev["nearest_y"],
        "V_x0_num": ev["V_interp"],
        "V_x0_exact": V_exact_x0,
        "relative_error_V_x0": abs(ev["V_interp"] - V_exact_x0) / abs(V_exact_x0),
        "pi_x0_num": ev["pi_interp"],
        "pi_x0_exact": pi_exact_x0,
        "relative_error_pi_x0": abs(ev["pi_interp"] - pi_exact_x0) / abs(pi_exact_x0),
        "control_RMSE_interior": float(np.sqrt(np.mean(control_error ** 2))),
        "control_relative_RMSE_interior": float(
            np.sqrt(np.mean(control_error ** 2)) / np.sqrt(np.mean(pi_exact_grid[mask] ** 2))
        ),
        "control_max_abs_error_interior": float(np.max(np.abs(control_error))),
        "fraction_final_Vxx_ge_0": float(np.mean(Vxx >= 0.0)),
        "n_final_Vxx_ge_0": int(np.sum(Vxx >= 0.0)),
        "runtime_seconds": runtime,
        **stats,
        **pi_stats,
    }
    return row, y, w, pi_num


def run_sweeps(config):
    space_rows = []
    time_rows = []
    profile_rows = []

    for gamma in GAMMAS:
        model = model_for_gamma(gamma)
        for Nx in NX_LIST:
            row, _, _, _ = run_one(model, Nx=Nx, Nt=config["space_Nt"], x0=config["x0"], eps=config["eps"])
            row["sweep"] = "Nx"
            space_rows.append(row)

        for Nt in NT_LIST:
            row, _, _, _ = run_one(model, Nx=config["time_Nx"], Nt=Nt, x0=config["x0"], eps=config["eps"])
            row["sweep"] = "Nt"
            time_rows.append(row)

        row, y, w, pi_num = run_one(
            model,
            Nx=config["profile_Nx"],
            Nt=config["profile_Nt"],
            x0=config["x0"],
            eps=config["eps"],
        )
        row["sweep"] = "profile"
        profile_rows.append(row)
        save_profile_arrays(model, y, w, pi_num)

    return pd.DataFrame(space_rows), pd.DataFrame(time_rows), pd.DataFrame(profile_rows)


def save_profile_arrays(model, y, w, pi_num):
    x = np.exp(y)
    Vx, Vxx, _ = derivatives_from_log_grid(y, w)
    df = pd.DataFrame(
        {
            "gamma": model.gamma,
            "y": y,
            "x": x,
            "V_num_t0": w,
            "V_exact_t0": model.V(0.0, x),
            "Vx_num_t0": Vx,
            "Vxx_num_t0": Vxx,
            "pi_num_t0": pi_num,
            "pi_exact_t0": model.pi_amount(x),
        }
    )
    write_csv(RESULT_DIR / f"profile_gamma_{model.gamma:g}.csv", df)


def plot_profiles():
    for gamma in GAMMAS:
        df = pd.read_csv(RESULT_DIR / f"profile_gamma_{gamma:g}.csv")
        mask = (df["x"] >= 0.25) & (df["x"] <= 4.0)

        plt.figure()
        plt.plot(df.loc[mask, "x"], df.loc[mask, "V_num_t0"], label="HJB FDM")
        plt.plot(df.loc[mask, "x"], df.loc[mask, "V_exact_t0"], "--", label="exact")
        plt.xlabel("x")
        plt.ylabel("V(0,x)")
        plt.title(f"Value profile, gamma={gamma:g}")
        plt.grid(True)
        plt.legend()
        plt.savefig(FIG_DIR / f"value_num_vs_exact_gamma_{gamma:g}.png", dpi=160, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(df.loc[mask, "x"], df.loc[mask, "pi_num_t0"], label="HJB FDM")
        plt.plot(df.loc[mask, "x"], df.loc[mask, "pi_exact_t0"], "--", label="exact")
        plt.xlabel("x")
        plt.ylabel("pi(0,x)")
        plt.title(f"Control profile, gamma={gamma:g}")
        plt.grid(True)
        plt.legend()
        plt.savefig(FIG_DIR / f"control_num_vs_exact_gamma_{gamma:g}.png", dpi=160, bbox_inches="tight")
        plt.close()


def plot_sweeps(space_df, time_df):
    plt.figure()
    for gamma, df in space_df.groupby("gamma"):
        plt.loglog(df["Nx"], df["relative_error_V_x0"], marker="o", label=f"gamma={gamma:g}")
    plt.xlabel("Nx")
    plt.ylabel("relative error on V(0,x0)")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(FIG_DIR / "error_value_vs_Nx.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    for gamma, df in time_df.groupby("gamma"):
        plt.loglog(df["Nt"], df["relative_error_V_x0"], marker="o", label=f"gamma={gamma:g}")
    plt.xlabel("Nt")
    plt.ylabel("relative error on V(0,x0)")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(FIG_DIR / "error_value_vs_Nt.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    for gamma, df in space_df.groupby("gamma"):
        plt.loglog(df["Nx"], df["control_relative_RMSE_interior"], marker="o", label=f"Nx gamma={gamma:g}")
    for gamma, df in time_df.groupby("gamma"):
        plt.loglog(df["Nt"], df["control_relative_RMSE_interior"], marker="s", linestyle="--", label=f"Nt gamma={gamma:g}")
    plt.xlabel("refinement parameter")
    plt.ylabel("relative RMSE control on interior grid")
    plt.grid(True, which="both")
    plt.legend(ncol=2, fontsize=8)
    plt.savefig(FIG_DIR / "error_control_vs_refinement.png", dpi=160, bbox_inches="tight")
    plt.close()


def make_summary(space_df, time_df, profile_df):
    best_by_gamma = []
    for gamma in GAMMAS:
        s = space_df[space_df["gamma"].eq(gamma)].sort_values("Nx")
        t = time_df[time_df["gamma"].eq(gamma)].sort_values("Nt")
        p = profile_df[profile_df["gamma"].eq(gamma)].iloc[0]
        best_by_gamma.append(
            {
                "gamma": gamma,
                "best_space_relative_error_V_x0": float(s["relative_error_V_x0"].min()),
                "best_time_relative_error_V_x0": float(t["relative_error_V_x0"].min()),
                "profile_relative_error_V_x0": float(p["relative_error_V_x0"]),
                "profile_control_RMSE_interior": float(p["control_RMSE_interior"]),
                "profile_control_max_abs_error_interior": float(p["control_max_abs_error_interior"]),
                "profile_fraction_final_Vxx_ge_0": float(p["fraction_final_Vxx_ge_0"]),
                "profile_fraction_denom_clipped_time_loop": float(p["fraction_denom_clipped"]),
                "space_error_decreases_monotonically": bool(
                    np.all(np.diff(s["relative_error_V_x0"].to_numpy()) <= 0.0)
                ),
                "time_error_decreases_monotonically": bool(
                    np.all(np.diff(t["relative_error_V_x0"].to_numpy()) <= 0.0)
                ),
            }
        )
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(RESULT_DIR),
        "n_space_rows": int(len(space_df)),
        "n_time_rows": int(len(time_df)),
        "n_profile_rows": int(len(profile_df)),
        "n_figures": int(len(list(FIG_DIR.glob("*.png")))),
        "by_gamma": best_by_gamma,
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "model": {
            "T": 1.0,
            "x0": 1.0,
            "r": 0.02,
            "mu": 0.08,
            "sigma": 0.20,
            "gammas": GAMMAS,
        },
        "numerics": {
            "state_variable": "y=log(x)",
            "y_min": -3.0,
            "y_max": 3.0,
            "Nx_list": NX_LIST,
            "Nt_list": NT_LIST,
            "space_Nt": 1600,
            "time_Nx": 200,
            "profile_Nx": 401,
            "profile_Nt": 1600,
            "eps": 1e-10,
            "interior_trim_fraction_each_side": 0.10,
        },
    }
    flat_config = {**config["numerics"], "x0": config["model"]["x0"]}

    write_json(RESULT_DIR / "config.json", config)
    space_df, time_df, profile_df = run_sweeps(flat_config)
    write_csv(RESULT_DIR / "sweep_Nx.csv", space_df)
    write_csv(RESULT_DIR / "sweep_Nt.csv", time_df)
    write_csv(RESULT_DIR / "profile_metrics.csv", profile_df)

    plot_profiles()
    plot_sweeps(space_df, time_df)
    summary = make_summary(space_df, time_df, profile_df)
    write_json(RESULT_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
