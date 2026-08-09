"""
Experience 2 : resolution de la HJB de Merton par differences finies.

On part de
    V_t + sup_pi [ (r x + pi(mu-r)) V_x + 0.5 pi^2 sigma^2 V_xx ] = 0,   V(T,x) = x^p/p.

Le sup interne (concave, V_xx<0) se resout explicitement :
    pi*(t,x) = -(mu-r)/sigma^2 * V_x/V_xx
    sup-term = -(mu-r)^2 V_x^2 / (2 sigma^2 V_xx)

d'ou la HJB "resolue" (encore non lineaire a cause du ratio V_x^2/V_xx) :
    V_t + r x V_x - (mu-r)^2 V_x^2 / (2 sigma^2 V_xx) = 0.

Changement de variable y = ln(x) (pour eviter le bord x=0 et les
non-uniformites d'une grille en x) : w(t,y) := V(t, e^y). Alors
    V_x  = w_y e^{-y}
    V_xx = (w_yy - w_y) e^{-2y}
et la HJB devient, en temps retourne tau = T - t (schema explicite backward) :
    w_tau = r w_y - (mu-r)^2 w_y^2 / (2 sigma^2 (w_yy - w_y)).

Conditions au bord (Dirichlet) : on impose la solution fermee de Part I aux
deux bords du domaine en y -- pratique standard pour un exercice de
validation ou la solution exacte est connue (pour un vrai probleme inconnu,
il faudrait des conditions asymptotiques ou de Neumann).
"""
import numpy as np
from merton_closed_form import MertonModel


def _empty_concavity_stats():
    return {
        "n_concavity_violations": 0,
        "n_near_zero_denom": 0,
        "n_denom_clipped": 0,
        "n_denom_points": 0,
        "min_denom": np.inf,
        "max_denom": -np.inf,
        "has_nan": False,
        "has_inf": False,
        "first_violation_time": np.nan,
    }


def _update_concavity_stats(stats, denom, t, eps):
    finite = denom[np.isfinite(denom)]
    if finite.size:
        stats["min_denom"] = min(stats["min_denom"], float(np.min(finite)))
        stats["max_denom"] = max(stats["max_denom"], float(np.max(finite)))
    stats["has_nan"] = bool(stats["has_nan"] or np.isnan(denom).any())
    stats["has_inf"] = bool(stats["has_inf"] or np.isinf(denom).any())

    n_viol = int(np.sum(denom >= 0.0))
    n_near = int(np.sum(np.abs(denom) < eps))
    stats["n_denom_points"] += int(denom.size)
    stats["n_concavity_violations"] += n_viol
    stats["n_near_zero_denom"] += n_near
    stats["n_denom_clipped"] += int(np.sum(denom >= -eps))
    if n_viol and np.isnan(stats["first_violation_time"]):
        stats["first_violation_time"] = float(t)


def terminal_condition(model: MertonModel, y):
    """Condition terminale w(T,y) = U(exp(y))."""
    return model.V(model.T, np.exp(y))


def solve_hjb_fdm(model: MertonModel, Ny=200, Nt=400, y_min=-3.0, y_max=3.0,
                  return_diagnostics=False, eps=1e-10):
    """Retourne (y_grid, w) avec w[i] = V(0, exp(y_i)) apres integration."""
    r, mu, sigma, T = model.r, model.mu, model.sigma, model.T
    y = np.linspace(y_min, y_max, Ny)
    dy = y[1] - y[0]
    dtau = T / Nt
    stats = _empty_concavity_stats()

    # condition terminale (tau=0  <=>  t=T)
    w = terminal_condition(model, y)  # w(y, tau=0)

    for n in range(Nt):
        tau = n * dtau
        t_now = T - tau
        w_y = np.empty_like(w)
        w_yy = np.empty_like(w)
        w_y[1:-1] = (w[2:] - w[:-2]) / (2 * dy)
        w_yy[1:-1] = (w[2:] - 2 * w[1:-1] + w[:-2]) / dy ** 2

        denom = w_yy[1:-1] - w_y[1:-1]
        _update_concavity_stats(stats, denom, t_now, eps)
        # Le denominateur doit rester negatif. En cas de denominateur positif
        # ou trop proche de zero, on clippe vers -eps et on le journalise.
        denom_safe = np.where(denom < -eps, denom, -eps)

        dw = r * w_y[1:-1] - (mu - r) ** 2 * w_y[1:-1] ** 2 / (2 * sigma ** 2 * denom_safe)
        w_new = w.copy()
        w_new[1:-1] = w[1:-1] + dtau * dw

        # bords : valeur exacte a t_now+dtau (le pas qu'on vient de calculer)
        t_next = T - (tau + dtau)
        w_new[0] = model.V(t_next, np.exp(y[0]))
        w_new[-1] = model.V(t_next, np.exp(y[-1]))

        w = w_new

    stats["has_nan"] = bool(stats["has_nan"] or np.isnan(w).any())
    stats["has_inf"] = bool(stats["has_inf"] or np.isinf(w).any())
    if np.isinf(stats["min_denom"]):
        stats["min_denom"] = np.nan
    if np.isinf(stats["max_denom"]):
        stats["max_denom"] = np.nan
    if stats["n_denom_points"]:
        stats["fraction_concavity_violations"] = (
            stats["n_concavity_violations"] / stats["n_denom_points"]
        )
        stats["fraction_denom_clipped"] = stats["n_denom_clipped"] / stats["n_denom_points"]
    else:
        stats["fraction_concavity_violations"] = np.nan
        stats["fraction_denom_clipped"] = np.nan
    stats["dt"] = float(dtau)
    stats["dy"] = float(dy)
    stats["lambda"] = float(dtau / dy ** 2)

    if return_diagnostics:
        return y, w, stats
    return y, w


def derivatives_from_log_grid(y, w):
    """Calcule V_x, V_xx et le denominateur log w_yy - w_y sur la grille."""
    dy = y[1] - y[0]
    w_y = np.gradient(w, dy)
    w_yy = np.gradient(w_y, dy)
    Vx = w_y * np.exp(-y)
    Vxx = (w_yy - w_y) * np.exp(-2 * y)
    return Vx, Vxx, w_yy - w_y


def pi_from_grid(model: MertonModel, y, w, eps=1e-10, return_diagnostics=False):
    """Reconstruit pi*(0,x) avec un denominateur negatif regularise si necessaire."""
    Vx, Vxx, _ = derivatives_from_log_grid(y, w)
    denom_safe = np.where(Vxx < -eps, Vxx, -eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        pi_amount = -(model.mu - model.r) / model.sigma ** 2 * Vx / Vxx
    pi_amount_safe = -(model.mu - model.r) / model.sigma ** 2 * Vx / denom_safe
    if return_diagnostics:
        stats = {
            "n_final_grid_points": int(Vxx.size),
            "n_final_concavity_violations": int(np.sum(Vxx >= 0.0)),
            "fraction_final_concavity_violations": float(np.mean(Vxx >= 0.0)),
            "n_final_denom_clipped": int(np.sum(Vxx >= -eps)),
            "fraction_final_denom_clipped": float(np.mean(Vxx >= -eps)),
            "min_final_Vxx": float(np.nanmin(Vxx)),
            "max_final_Vxx": float(np.nanmax(Vxx)),
            "has_nan_pi_raw": bool(np.isnan(pi_amount).any()),
            "has_inf_pi_raw": bool(np.isinf(pi_amount).any()),
            "has_nan_pi": bool(np.isnan(pi_amount_safe).any()),
            "has_inf_pi": bool(np.isinf(pi_amount_safe).any()),
        }
        return pi_amount_safe, Vx, Vxx, stats
    return pi_amount_safe


def interpolate_at_y(y, values, y0):
    """Interpolation lineaire sur la grille y."""
    return float(np.interp(y0, y, values))


def evaluate_solution_at_x0(model: MertonModel, y, w, x0=1.0):
    y0 = np.log(x0)
    pi_grid = pi_from_grid(model, y, w)
    i0 = int(np.argmin(np.abs(y - y0)))
    return {
        "y0": float(y0),
        "nearest_y": float(y[i0]),
        "V_nearest": float(w[i0]),
        "pi_nearest": float(pi_grid[i0]),
        "V_interp": interpolate_at_y(y, w, y0),
        "pi_interp": interpolate_at_y(y, pi_grid, y0),
    }


def convergence_study(model: MertonModel, x0=1.0,
                       Ny_list=(50, 100, 200, 400, 800),
                       Nt_list=(50, 100, 200, 400, 800)):
    """Fait varier Ny et Nt independamment (l'autre parametre fixe a une valeur fine),
    mesure eps_V et eps_pi au point (t=0, x=x0)."""
    y0 = np.log(x0)
    V_exact = model.V(0.0, x0)
    pi_exact = model.pi_amount(x0)

    rows = []
    Nt_fixed = 800
    for Ny in Ny_list:
        y, w = solve_hjb_fdm(model, Ny=Ny, Nt=Nt_fixed)
        eval0 = evaluate_solution_at_x0(model, y, w, x0=x0)
        V_hat = eval0["V_interp"]
        pi_hat = eval0["pi_interp"]
        eps_V = abs(V_hat - V_exact) / abs(V_exact)
        eps_pi = abs(pi_hat - pi_exact) / abs(pi_exact)
        rows.append({"sweep": "Ny", "Ny": Ny, "Nt": Nt_fixed, "eps_V": eps_V, "eps_pi": eps_pi})

    Ny_fixed = 800
    for Nt in Nt_list:
        y, w = solve_hjb_fdm(model, Ny=Ny_fixed, Nt=Nt)
        eval0 = evaluate_solution_at_x0(model, y, w, x0=x0)
        V_hat = eval0["V_interp"]
        pi_hat = eval0["pi_interp"]
        eps_V = abs(V_hat - V_exact) / abs(V_exact)
        eps_pi = abs(pi_hat - pi_exact) / abs(pi_exact)
        rows.append({"sweep": "Nt", "Ny": Ny_fixed, "Nt": Nt, "eps_V": eps_V, "eps_pi": eps_pi})

    return rows


if __name__ == "__main__":
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)

    y, w = solve_hjb_fdm(model, Ny=400, Nt=800)
    x0 = 1.0
    eval0 = evaluate_solution_at_x0(model, y, w, x0=x0)
    V_hat = eval0["V_interp"]
    V_exact = model.V(0.0, x0)
    pi_hat = eval0["pi_interp"]
    pi_exact = model.pi_amount(x0)

    print(f"V(0,1)   FDM={V_hat:.6f}   exact={V_exact:.6f}   err={abs(V_hat-V_exact)/abs(V_exact):.4%}")
    print(f"pi*(0,1) FDM={pi_hat:.6f}   exact={pi_exact:.6f}   err={abs(pi_hat-pi_exact)/abs(pi_exact):.4%}")

    print("\n--- Etude de convergence ---")
    rows = convergence_study(model, x0=1.0)
    for row in rows:
        print(row)
