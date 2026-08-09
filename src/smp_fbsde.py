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


def run_smp_fbsde(model: MertonModel, N=50, M=40_000, x0=1.0, seed=0):
    r, mu, sigma, T, p = model.r, model.mu, model.sigma, model.T, model.p
    dt = T / N
    rng = np.random.default_rng(seed)
    pi_star = model.pi_star  # fraction constante

    dW = rng.normal(scale=np.sqrt(dt), size=(M, N))
    X = np.empty((M, N + 1))
    X[:, 0] = x0
    drift = r + pi_star * (mu - r)
    for i in range(N):
        X[:, i + 1] = X[:, i] * (1 + drift * dt + pi_star * sigma * dW[:, i])
    X = np.maximum(X, 1e-8)

    # --- regression backward pour Y_t = p_t = V_x(t, X*_t) ---
    Y = model.p * X[:, -1] ** (p - 1) / model.p * model.p  # = X_T^(p-1) (terminal: p_T = U'(X_T) = X_T^{p-1})
    Y = X[:, -1] ** (p - 1)
    basis = lambda x: np.column_stack([np.ones_like(x), x ** (p - 1)])

    t_grid = np.linspace(0, T, N + 1)
    phi_hat = np.empty(N + 1)
    phi_hat[N] = 1.0
    Z_estimates = np.empty(N)  # q_t estimated at each step (mean over paths, for diagnostics)

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

    return t_grid, X, phi_hat, Z_estimates


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
    pi_smp = -(mu - r) / sigma ** 2 * p_bsde / Vxx_exact if Vxx_exact != 0 else np.nan
    pi_exact_amount = model.pi_amount(np.mean(Xt))
    err_pi = abs(pi_smp - pi_exact_amount)

    return {
        "t": t,
        "p_bsde": p_bsde, "Vx_exact": Vx_exact, "err_p_vs_Vx": err_p_vs_Vx,
        "q_bsde": q_bsde, "q_theory": q_theory_mean, "err_q_vs_theory": err_q_vs_theory,
        "H_pi": H_pi,
        "pi_smp": pi_smp, "pi_exact": pi_exact_amount, "err_pi": err_pi,
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
