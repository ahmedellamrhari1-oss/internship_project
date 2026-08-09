import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hjb_fdm import evaluate_solution_at_x0, solve_hjb_fdm  # noqa: E402
from merton_closed_form import MertonModel  # noqa: E402
from smp_fbsde import simulate_merton_gbm_exact  # noqa: E402


def assert_close(value, target, tol, name):
    if not abs(value - target) <= tol:
        raise AssertionError(f"{name}: {value} not within {tol} of {target}")


def test_merton_identity():
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    x = np.array([0.5, 1.0, 2.0])
    t = 0.3
    pi_from_derivatives = -(model.mu - model.r) / model.sigma ** 2 * model.Vx(t, x) / model.Vxx(t, x)
    np.testing.assert_allclose(pi_from_derivatives, model.pi_amount(x), rtol=1e-12, atol=1e-12)


def test_log_transform_vxx():
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    y = np.linspace(-1.0, 1.0, 2001)
    dy = y[1] - y[0]
    t = 0.4
    w = model.V(t, np.exp(y))
    w_y = np.gradient(w, dy)
    w_yy = np.gradient(w_y, dy)
    Vxx_from_y = (w_yy - w_y) * np.exp(-2 * y)
    Vxx_exact = model.Vxx(t, np.exp(y))
    np.testing.assert_allclose(Vxx_from_y[10:-10], Vxx_exact[10:-10], rtol=2e-4, atol=2e-4)


def test_interpolation_at_y0_matches_odd_node():
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    y, w = solve_hjb_fdm(model, Ny=101, Nt=400)
    ev = evaluate_solution_at_x0(model, y, w, x0=1.0)
    assert_close(ev["nearest_y"], 0.0, 1e-14, "nearest_y")
    assert_close(ev["V_nearest"], ev["V_interp"], 1e-14, "V interpolation at node")
    assert_close(ev["pi_nearest"], ev["pi_interp"], 1e-14, "pi interpolation at node")


def test_exact_gbm_positivity():
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    X, _ = simulate_merton_gbm_exact(model, N=20, M=2000, x0=1.0, seed=123)
    if not np.all(X > 0):
        raise AssertionError("Exact GBM simulation produced non-positive wealth")


def test_analytic_smp_identity():
    model = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    x = np.array([0.7, 1.0, 1.8])
    t = 0.6
    p = model.Vx(t, x)
    q = model.sigma * model.pi_amount(x) * model.Vxx(t, x)
    H_pi = p * (model.mu - model.r) + q * model.sigma
    np.testing.assert_allclose(H_pi, 0.0, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    test_merton_identity()
    test_log_transform_vxx()
    test_interpolation_at_y0_matches_odd_node()
    test_exact_gbm_positivity()
    test_analytic_smp_identity()
    print("All core tests passed.")
