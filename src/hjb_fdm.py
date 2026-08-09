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


def solve_hjb_fdm(model: MertonModel, Ny=200, Nt=400, y_min=-3.0, y_max=3.0):
    """Retourne (y_grid, t_grid, w) avec w[i,j] = V(t_j, exp(y_i))."""
    r, mu, sigma, T = model.r, model.mu, model.sigma, model.T
    y = np.linspace(y_min, y_max, Ny)
    dy = y[1] - y[0]
    dtau = T / Nt

    # condition terminale (tau=0  <=>  t=T)
    w = model.V(T, np.exp(y))  # w(y, tau=0)

    for n in range(Nt):
        tau = n * dtau
        t_now = T - tau
        w_y = np.empty_like(w)
        w_yy = np.empty_like(w)
        w_y[1:-1] = (w[2:] - w[:-2]) / (2 * dy)
        w_yy[1:-1] = (w[2:] - 2 * w[1:-1] + w[:-2]) / dy ** 2

        denom = w_yy[1:-1] - w_y[1:-1]
        # garde-fou numerique : le denominateur doit rester du signe de V_xx (concavite)
        eps = 1e-10
        denom_safe = np.where(np.abs(denom) < eps, np.sign(denom) * eps + eps, denom)

        dw = r * w_y[1:-1] - (mu - r) ** 2 * w_y[1:-1] ** 2 / (2 * sigma ** 2 * denom_safe)
        w_new = w.copy()
        w_new[1:-1] = w[1:-1] + dtau * dw

        # bords : valeur exacte a t_now+dtau (le pas qu'on vient de calculer)
        t_next = T - (tau + dtau)
        w_new[0] = model.V(t_next, np.exp(y[0]))
        w_new[-1] = model.V(t_next, np.exp(y[-1]))

        w = w_new

    return y, w


def pi_from_grid(model: MertonModel, y, w):
    """pi*(0, x) = -(mu-r)/sigma^2 * V_x/V_xx, calcule par differences finies sur la grille finale."""
    dy = y[1] - y[0]
    w_y = np.gradient(w, dy)
    w_yy = np.gradient(w_y, dy)
    x = np.exp(y)
    Vx = w_y * np.exp(-y)
    Vxx = (w_yy - w_y) * np.exp(-2 * y)
    pi_amount = -(model.mu - model.r) / model.sigma ** 2 * Vx / Vxx
    return pi_amount


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
        i0 = np.argmin(np.abs(y - y0))
        V_hat = w[i0]
        pi_hat = pi_from_grid(model, y, w)[i0]
        eps_V = abs(V_hat - V_exact) / abs(V_exact)
        eps_pi = abs(pi_hat - pi_exact) / abs(pi_exact)
        rows.append({"sweep": "Ny", "Ny": Ny, "Nt": Nt_fixed, "eps_V": eps_V, "eps_pi": eps_pi})

    Ny_fixed = 800
    for Nt in Nt_list:
        y, w = solve_hjb_fdm(model, Ny=Ny_fixed, Nt=Nt)
        i0 = np.argmin(np.abs(y - y0))
        V_hat = w[i0]
        pi_hat = pi_from_grid(model, y, w)[i0]
        eps_V = abs(V_hat - V_exact) / abs(V_exact)
        eps_pi = abs(pi_hat - pi_exact) / abs(pi_exact)
        rows.append({"sweep": "Nt", "Ny": Ny_fixed, "Nt": Nt, "eps_V": eps_V, "eps_pi": eps_pi})

    return rows


if __name__ == "__main__":
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)

    y, w = solve_hjb_fdm(model, Ny=400, Nt=800)
    x0 = 1.0
    i0 = np.argmin(np.abs(y - np.log(x0)))
    V_hat = w[i0]
    V_exact = model.V(0.0, x0)
    pi_hat = pi_from_grid(model, y, w)[i0]
    pi_exact = model.pi_amount(x0)

    print(f"V(0,1)   FDM={V_hat:.6f}   exact={V_exact:.6f}   err={abs(V_hat-V_exact)/abs(V_exact):.4%}")
    print(f"pi*(0,1) FDM={pi_hat:.6f}   exact={pi_exact:.6f}   err={abs(pi_hat-pi_exact)/abs(pi_exact):.4%}")

    print("\n--- Etude de convergence ---")
    rows = convergence_study(model, x0=1.0)
    for row in rows:
        print(row)