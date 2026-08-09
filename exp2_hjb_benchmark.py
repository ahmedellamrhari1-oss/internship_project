import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from hjb_fdm import evaluate_solution_at_x0, solve_hjb_fdm  # noqa: E402
from merton_closed_form import MertonModel  # noqa: E402


RESULT_DIR = Path("results") / "exp2"
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


def empirical_orders(errors, hs):
    orders = [np.nan]
    for i in range(1, len(errors)):
        if errors[i - 1] > 0 and errors[i] > 0 and hs[i - 1] != hs[i]:
            orders.append(float(np.log(errors[i - 1] / errors[i]) / np.log(hs[i - 1] / hs[i])))
        else:
            orders.append(np.nan)
    return orders


def run_space_sweep(model, x0, Ny_list, Nt):
    rows = []
    V_exact = model.V(0.0, x0)
    pi_exact = model.pi_amount(x0)
    for Ny in Ny_list:
        y, w, stats = solve_hjb_fdm(model, Ny=Ny, Nt=Nt, return_diagnostics=True)
        ev = evaluate_solution_at_x0(model, y, w, x0=x0)
        rel_V = abs(ev["V_interp"] - V_exact) / abs(V_exact)
        rel_pi = abs(ev["pi_interp"] - pi_exact) / abs(pi_exact)
        rows.append({
            "Ny": Ny,
            "Nt": Nt,
            "dy": stats["dy"],
            "dt": stats["dt"],
            "lambda": stats["lambda"],
            "V_hat": ev["V_interp"],
            "V_exact": V_exact,
            "pi_hat": ev["pi_interp"],
            "pi_exact": pi_exact,
            "relative_error_V": rel_V,
            "relative_error_pi": rel_pi,
            **stats,
        })
    rows = sorted(rows, key=lambda r: r["dy"], reverse=True)
    orders_V = empirical_orders([r["relative_error_V"] for r in rows], [r["dy"] for r in rows])
    orders_pi = empirical_orders([r["relative_error_pi"] for r in rows], [r["dy"] for r in rows])
    for row, order_v, order_pi in zip(rows, orders_V, orders_pi):
        row["order_V"] = order_v
        row["order_pi"] = order_pi
    return rows


def run_time_sweep(model, x0, Ny, Nt_list):
    rows = []
    V_exact = model.V(0.0, x0)
    pi_exact = model.pi_amount(x0)
    for Nt in Nt_list:
        y, w, stats = solve_hjb_fdm(model, Ny=Ny, Nt=Nt, return_diagnostics=True)
        ev = evaluate_solution_at_x0(model, y, w, x0=x0)
        rows.append({
            "Ny": Ny,
            "Nt": Nt,
            "dt": stats["dt"],
            "dy": stats["dy"],
            "lambda": stats["lambda"],
            "V_hat": ev["V_interp"],
            "V_exact": V_exact,
            "pi_hat": ev["pi_interp"],
            "pi_exact": pi_exact,
            "relative_error_V": abs(ev["V_interp"] - V_exact) / abs(V_exact),
            "relative_error_pi": abs(ev["pi_interp"] - pi_exact) / abs(pi_exact),
            **stats,
        })
    return rows


def run_evaluation_comparison(model, x0, Nt):
    rows = []
    V_exact = model.V(0.0, x0)
    pi_exact = model.pi_amount(x0)
    for Ny in (800, 801):
        y, w, stats = solve_hjb_fdm(model, Ny=Ny, Nt=Nt, return_diagnostics=True)
        ev = evaluate_solution_at_x0(model, y, w, x0=x0)
        rows.append({
            "Ny": Ny,
            "Nt": Nt,
            "dy": stats["dy"],
            "y0": ev["y0"],
            "nearest_y": ev["nearest_y"],
            "V_nearest": ev["V_nearest"],
            "V_interp": ev["V_interp"],
            "pi_nearest": ev["pi_nearest"],
            "pi_interp": ev["pi_interp"],
            "relative_error_V_nearest": abs(ev["V_nearest"] - V_exact) / abs(V_exact),
            "relative_error_V_interp": abs(ev["V_interp"] - V_exact) / abs(V_exact),
            "relative_error_pi_nearest": abs(ev["pi_nearest"] - pi_exact) / abs(pi_exact),
            "relative_error_pi_interp": abs(ev["pi_interp"] - pi_exact) / abs(pi_exact),
        })
    return rows


def plot_results(space_df, time_df):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIG_DIR.glob("*.png"):
        archive_if_exists(path)

    plt.figure()
    plt.loglog(space_df["Ny"], space_df["relative_error_V"], marker="o")
    plt.xlabel("Ny")
    plt.ylabel("relative error V")
    plt.grid(True, which="both")
    plt.savefig(FIG_DIR / "error_V_vs_Ny.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.loglog(space_df["Ny"], space_df["relative_error_pi"], marker="o")
    plt.xlabel("Ny")
    plt.ylabel("relative error pi")
    plt.grid(True, which="both")
    plt.savefig(FIG_DIR / "error_pi_vs_Ny.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(time_df["Nt"], time_df["relative_error_V"], marker="o", label="V")
    plt.plot(time_df["Nt"], time_df["relative_error_pi"], marker="o", label="pi")
    plt.xlabel("Nt")
    plt.ylabel("relative error")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(FIG_DIR / "error_V_pi_vs_Nt.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(time_df["lambda"], time_df["relative_error_V"], marker="o", label="V")
    plt.plot(time_df["lambda"], time_df["relative_error_pi"], marker="o", label="pi")
    plt.xlabel("lambda = dt / dy^2")
    plt.ylabel("relative error")
    plt.yscale("log")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(FIG_DIR / "error_V_pi_vs_lambda.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(time_df["lambda"], time_df["n_concavity_violations"], marker="o")
    plt.xlabel("lambda = dt / dy^2")
    plt.ylabel("concavity violations")
    plt.grid(True)
    plt.savefig(FIG_DIR / "concavity_violations_vs_lambda.png", dpi=160, bbox_inches="tight")
    plt.close()


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    x0 = 1.0
    space_Ny = [51, 101, 201, 401, 801]
    space_Nt = 1200
    time_Ny = 801
    time_Nt = [40, 50, 60, 75, 90, 100, 110, 125, 150, 175, 200, 250, 300, 400, 600, 800, 1200]

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
        },
        "numerical_parameters": {
            "space_Ny": space_Ny,
            "space_Nt": space_Nt,
            "time_Ny": time_Ny,
            "time_Nt": time_Nt,
            "y_min": -3.0,
            "y_max": 3.0,
        },
    }

    comparison = pd.DataFrame(run_evaluation_comparison(model, x0, space_Nt))
    space_df = pd.DataFrame(run_space_sweep(model, x0, space_Ny, space_Nt))
    time_df = pd.DataFrame(run_time_sweep(model, x0, time_Ny, time_Nt))

    write_json(RESULT_DIR / "config.json", config)
    write_csv(RESULT_DIR / "evaluation_comparison.csv", comparison)
    write_csv(RESULT_DIR / "convergence_space.csv", space_df)
    write_csv(RESULT_DIR / "stability_time.csv", time_df)
    plot_results(space_df, time_df)

    summary = {
        "best_space_error_V": float(space_df["relative_error_V"].min()),
        "best_space_error_pi": float(space_df["relative_error_pi"].min()),
        "space_last_order_V": float(space_df["order_V"].dropna().iloc[-1]),
        "space_last_order_pi": float(space_df["order_pi"].dropna().iloc[-1]),
        "first_time_sweep_with_no_concavity_violations": (
            int(time_df.loc[time_df["n_concavity_violations"].eq(0), "Nt"].iloc[0])
            if time_df["n_concavity_violations"].eq(0).any() else None
        ),
        "max_lambda_without_concavity_violations": (
            float(time_df.loc[time_df["n_concavity_violations"].eq(0), "lambda"].max())
            if time_df["n_concavity_violations"].eq(0).any() else None
        ),
        "n_figures": len(list(FIG_DIR.glob("*.png"))),
    }
    write_json(RESULT_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
