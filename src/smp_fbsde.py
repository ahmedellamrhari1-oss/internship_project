"""
Experience 3 : SMP / BSDE adjointe pour Merton, avec diagnostics complets.

Rappel du raisonnement correct :
  - HJB donne V(t,x) = phi(t) x^p / p  =>  p_t := V_x(t,X*_t) = phi(t) (X*_t)^(p-1)
  - Ito sur p_t = phi(t) (X*_t)^(p-1) donne le terme de diffusion :
        q_t = (p-1) phi(t) (X*_t)^(p-2) * X*_t * pi*_t * sigma / X*_t
            = (p-1) * V_xx(t,X*_t) * pi*_t * sigma / ... (voir derivation ci-dessous)
    Plus simplement, avec p_t = V_x et q_t = sigma * pi*_t * V_xx (Ito standard,
    puisque dX* = ... + pi* sigma dW et p_t=V_x(t,X*_t) => dp_t = ... + V_xx * pi* sigma dW_t)
  - La condition SMP p(mu-r) + q*sigma = 0 est bien verifiee car
        V_x (mu-r) + sigma pi* V_xx * sigma = V_x(mu-r) + pi* sigma^2 V_xx
    et pi* = -(mu-r)/sigma^2 * V_x/V_xx  =>  le terme s'annule exactement. C'est
    une IDENTITE (consequence de la definition de pi*), pas une equation qu'on
    resout pour pi* -- point que l'autre IA a corrige dans le rapport.

On simule ici la BSDE adjointe par un schema de regression (comme dans
fbsde_merton.py), puis on calcule TOUS les diagnostics demandes :
    || p_t - V_x(t,X*_t) ||          (BSDE adjointe vs formule HJB)
    || q_t - sigma * pi*_t * V_xx(t,X*_t) ||   (coherence Ito)
    || H_pi(t, X*_t, pi*_t, p_t, q_t) ||        (condition SMP, doit etre ~0)
    || pi_SMP - pi_exact ||
"""
import numpy as np
from merton_closed_form import MertonModel


def simulate_merton_gbm_exact(model: MertonModel, N=50, M=40_000, x0=1.0, seed=0):
    """Simule exactement la richesse optimale de Merton sous controle proportionnel."""
    r, mu, sigma, T, p = model.r, model.mu, model.sigma, model.T, model.p
    dt = T / N
    rng = np.random.default_rng(seed)
    pi_star = model.pi_star  # fraction constante

    dW = rng.normal(scale=np.sqrt(dt), size=(M, N))
    X = np.empty((M, N + 1))
    X[:, 0] = x0
    drift = r + pi_star * (mu - r)
    diffusion = pi_star * sigma
    for i in range(N):
        X[:, i + 1] = X[:, i] * np.exp((drift - 0.5 * diffusion ** 2) * dt + diffusion * dW[:, i])
    return X, dW


def simulate_merton_euler(model: MertonModel, N=50, M=40_000, x0=1.0, seed=0):
    """Ancien benchmark Euler avec floor, conserve pour comparaison si necessaire."""
    r, mu, sigma, T = model.r, model.mu, model.sigma, model.T
    dt = T / N
    rng = np.random.default_rng(seed)
    pi_star = model.pi_star
    dW = rng.normal(scale=np.sqrt(dt), size=(M, N))
    X = np.empty((M, N + 1))
    X[:, 0] = x0
    drift = r + pi_star * (mu - r)
    for i in range(N):
        X[:, i + 1] = X[:, i] * (1 + drift * dt + pi_star * sigma * dW[:, i])
    return np.maximum(X, 1e-8), dW


def _eval_indices_from_times(T, N, eval_times=None):
    if eval_times is None:
        eval_times = [0.0, 0.25 * T, 0.5 * T, 0.75 * T, T]
    idx = sorted(set(int(round(t / T * N)) for t in eval_times))
    return [min(max(i, 0), N) for i in idx]


def run_smp_fbsde(model: MertonModel, N=50, M=40_000, x0=1.0, seed=0,
                  scheme="exact", eval_times=None, return_pathwise=False):
    r, mu, sigma, T, p = model.r, model.mu, model.sigma, model.T, model.p
    dt = T / N
    pi_star = model.pi_star  # fraction constante

    if scheme == "exact":
        X, dW = simulate_merton_gbm_exact(model, N=N, M=M, x0=x0, seed=seed)
    elif scheme == "euler":
        X, dW = simulate_merton_euler(model, N=N, M=M, x0=x0, seed=seed)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    # --- regression backward pour Y_t = p_t = V_x(t, X*_t) ---
    Y = X[:, -1] ** (p - 1)
    basis = lambda x: np.column_stack([np.ones_like(x), x ** (p - 1)])

    t_grid = np.linspace(0, T, N + 1)
    phi_hat = np.empty(N + 1)
    phi_hat[N] = 1.0
    Z_estimates = np.empty(N)  # q_t estimated at each step (mean over paths, for diagnostics)
    eval_indices = _eval_indices_from_times(T, N, eval_times)
    p_hat_eval = {}
    q_hat_eval = {}
    if N in eval_indices:
        p_hat_eval[N] = Y.copy()
        q_hat_eval[N] = np.full(M, np.nan)

    for i in range(N - 1, -1, -1):
        Xi = X[:, i]
        B = basis(Xi)

        coef_y, *_ = np.linalg.lstsq(B, Y, rcond=None)
        cond_exp_Y = B @ coef_y

        coef_z, *_ = np.linalg.lstsq(B, Y * dW[:, i], rcond=None)
        cond_exp_Z = (B @ coef_z) / dt

        driver = (r + pi_star * (mu - r)) * cond_exp_Y + pi_star * sigma * cond_exp_Z
        Y = cond_exp_Y + driver * dt
        phi_hat[i] = np.mean(Y / (Xi ** (p - 1)))
        Z_estimates[i] = np.mean(cond_exp_Z)
        if i in eval_indices:
            p_hat_eval[i] = Y.copy()
            q_hat_eval[i] = cond_exp_Z.copy()

    if return_pathwise:
        return {
            "t_grid": t_grid,
            "X": X,
            "phi_hat": phi_hat,
            "Z_estimates": Z_estimates,
            "p_hat_eval": p_hat_eval,
            "q_hat_eval": q_hat_eval,
            "eval_indices": eval_indices,
            "scheme": scheme,
            "seed": seed,
            "N": N,
            "M": M,
            "dt": dt,
        }
    return t_grid, X, phi_hat, Z_estimates


def _error_summary(errors, scale=None):
    abs_err = np.abs(errors)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    denom = float(scale if scale is not None else 0.0)
    return {
        "rmse": rmse,
        "relative_rmse": float(rmse / denom) if denom > 0 else np.nan,
        "mae": float(np.mean(abs_err)),
        "median_abs_error": float(np.median(abs_err)),
        "q05_abs_error": float(np.quantile(abs_err, 0.05)),
        "q50_abs_error": float(np.quantile(abs_err, 0.50)),
        "q95_abs_error": float(np.quantile(abs_err, 0.95)),
    }


def pathwise_diagnostics(model: MertonModel, result):
    """Diagnostics pathwise aux temps stockes par run_smp_fbsde(..., return_pathwise=True)."""
    r, mu, sigma = model.r, model.mu, model.sigma
    rows = []
    t_grid = result["t_grid"]
    X = result["X"]
    alpha = model.pi_star

    for idx in result["eval_indices"]:
        t = t_grid[idx]
        Xt = X[:, idx]
        p_hat = result["p_hat_eval"][idx]
        q_hat = result["q_hat_eval"][idx]
        p_exact = model.Vx(t, Xt)
        pi_exact = alpha * Xt
        q_exact = sigma * pi_exact * model.Vxx(t, Xt)

        p_stats = _error_summary(p_hat - p_exact, np.sqrt(np.mean(p_exact ** 2)))
        row = {
            "t_index": idx,
            "t": float(t),
            "seed": result["seed"],
            "N": result["N"],
            "M": result["M"],
            "dt": result["dt"],
            "scheme": result["scheme"],
            "RMSE_p": p_stats["rmse"],
            "relative_RMSE_p": p_stats["relative_rmse"],
            "MAE_p": p_stats["mae"],
            "median_abs_error_p": p_stats["median_abs_error"],
            "q05_abs_error_p": p_stats["q05_abs_error"],
            "q50_abs_error_p": p_stats["q50_abs_error"],
            "q95_abs_error_p": p_stats["q95_abs_error"],
        }

        if np.isfinite(q_hat).all():
            q_stats = _error_summary(q_hat - q_exact, np.sqrt(np.mean(q_exact ** 2)))
            H_pi_hat_path = p_hat * (mu - r) + q_hat * sigma
            H_stats = _error_summary(H_pi_hat_path, 1.0)
            pi_reconstructed = -(mu - r) / sigma ** 2 * p_hat / model.Vxx(t, Xt)
            pi_stats = _error_summary(pi_reconstructed - pi_exact, np.sqrt(np.mean(pi_exact ** 2)))
            row.update({
                "RMSE_q": q_stats["rmse"],
                "relative_RMSE_q": q_stats["relative_rmse"],
                "MAE_q": q_stats["mae"],
                "median_abs_error_q": q_stats["median_abs_error"],
                "q05_abs_error_q": q_stats["q05_abs_error"],
                "q50_abs_error_q": q_stats["q50_abs_error"],
                "q95_abs_error_q": q_stats["q95_abs_error"],
                "RMSE_Hpi": H_stats["rmse"],
                "MAE_Hpi": H_stats["mae"],
                "median_abs_error_Hpi": H_stats["median_abs_error"],
                "q05_abs_error_Hpi": H_stats["q05_abs_error"],
                "q50_abs_error_Hpi": H_stats["q50_abs_error"],
                "q95_abs_error_Hpi": H_stats["q95_abs_error"],
                "RMSE_pi_reconstructed": pi_stats["rmse"],
                "relative_RMSE_pi_reconstructed": pi_stats["relative_rmse"],
                "MAE_pi_reconstructed": pi_stats["mae"],
                "median_abs_error_pi_reconstructed": pi_stats["median_abs_error"],
                "q05_abs_error_pi_reconstructed": pi_stats["q05_abs_error"],
                "q50_abs_error_pi_reconstructed": pi_stats["q50_abs_error"],
                "q95_abs_error_pi_reconstructed": pi_stats["q95_abs_error"],
            })
        else:
            row.update({
                "RMSE_q": np.nan,
                "relative_RMSE_q": np.nan,
                "RMSE_Hpi": np.nan,
                "RMSE_pi_reconstructed": np.nan,
                "relative_RMSE_pi_reconstructed": np.nan,
            })
        rows.append(row)
    return rows


def diagnostics(model: MertonModel, t_grid, X, phi_hat, Z_estimates, t_index=0):
    """Calcule les 4 diagnostics demandes au pas de temps t_index, moyennes sur les trajectoires."""
    r, mu, sigma, p = model.r, model.mu, model.sigma, model.p
    t = t_grid[t_index]
    Xt = X[:, t_index]
    pi_star = model.pi_star

    # p_t estime par la BSDE (regression) vs V_x exact (HJB)
    p_bsde = phi_hat[t_index] * np.mean(Xt ** (p - 1))
    Vx_exact = np.mean(model.Vx(t, Xt))
    err_p_vs_Vx = abs(p_bsde - Vx_exact)

    # q_t estime vs sigma * pi* * Vxx (identite d'Ito attendue)
    q_bsde = Z_estimates[t_index] if t_index < len(Z_estimates) else np.nan
    Vxx_exact = np.mean(model.Vxx(t, Xt))
    q_theory = sigma * pi_star * np.mean(Xt) * Vxx_exact / np.mean(Xt)  # sigma*pi_amount*Vxx, pi_amount=pi*·X
    # plus precis : q_theory_t = sigma * pi_amount_t * Vxx_t, moyenne sur trajectoires
    q_theory_mean = np.mean(sigma * (pi_star * Xt) * model.Vxx(t, Xt))
    err_q_vs_theory = abs(q_bsde - q_theory_mean) if not np.isnan(q_bsde) else np.nan

    # condition SMP : H_pi = p(mu-r) + q*sigma (doit etre ~0 a l'optimum)
    H_pi = p_bsde * (mu - r) + q_bsde * sigma if not np.isnan(q_bsde) else np.nan

    # pi_SMP reconstruit vs pi exact
    pi_reconstructed = -(mu - r) / sigma ** 2 * p_bsde / Vxx_exact if Vxx_exact != 0 else np.nan
    pi_exact_amount = model.pi_amount(np.mean(Xt))
    err_pi = abs(pi_reconstructed - pi_exact_amount)

    return {
        "t": t,
        "p_bsde": p_bsde, "Vx_exact": Vx_exact, "err_p_vs_Vx": err_p_vs_Vx,
        "q_bsde": q_bsde, "q_theory": q_theory_mean, "err_q_vs_theory": err_q_vs_theory,
        "H_pi": H_pi,
        "pi_reconstructed": pi_reconstructed, "pi_exact": pi_exact_amount, "err_pi": err_pi,
    }


if __name__ == "__main__":
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    t_grid, X, phi_hat, Z_estimates = run_smp_fbsde(model, N=50, M=40_000, x0=1.0, seed=0)

    print(f"phi(0) BSDE = {phi_hat[0]:.6f}   phi(0) closed-form = {model.phi(0.0):.6f}\n")

    diag0 = diagnostics(model, t_grid, X, phi_hat, Z_estimates, t_index=0)
    print("--- Diagnostics a t=0 ---")
    for k, v in diag0.items():
        print(f"  {k:16s} = {v:.6f}" if isinstance(v, (float, np.floating)) else f"  {k:16s} = {v}")

    diag_mid = diagnostics(model, t_grid, X, phi_hat, Z_estimates, t_index=25)
    print("\n--- Diagnostics a t=T/2 ---")
    for k, v in diag_mid.items():
        print(f"  {k:16s} = {v:.6f}" if isinstance(v, (float, np.floating)) else f"  {k:16s} = {v}")
