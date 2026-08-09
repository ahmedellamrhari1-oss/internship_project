"""
Experience 1 : Merton 1D, solution analytique (ground truth).

Convention utilisee dans tout le projet : utilite CRRA U(x) = x^p / p avec
p = 1 - gamma (gamma = aversion relative au risque, gamma > 0, gamma != 1).

Rapport (chapitre Merton) : U(x) = x^(1-gamma)/(1-gamma)  =>  p = 1 - gamma.
    pi*_t = (mu - r) / (gamma * sigma^2) * X_t        (montant investi)
    V(t,x) = exp(rho (T-t)) * x^p / p
    rho    = r*p + (mu-r)^2 / (2*sigma^2) * p/(1-p)
"""
import numpy as np


class MertonModel:
    def __init__(self, r, mu, sigma, gamma, T):
        self.r, self.mu, self.sigma, self.gamma, self.T = r, mu, sigma, gamma, T
        self.p = 1.0 - gamma
        self.rho = r * self.p + (mu - r) ** 2 / (2 * sigma ** 2) * self.p / (1 - self.p)
        self.pi_star = (mu - r) / (gamma * sigma ** 2)          # proportion (fraction de richesse)

    def phi(self, t):
        return np.exp(self.rho * (self.T - t))

    def V(self, t, x):
        """Fonction de valeur V(t,x) = phi(t) x^p / p."""
        return self.phi(t) * x ** self.p / self.p

    def Vx(self, t, x):
        return self.phi(t) * x ** (self.p - 1)

    def Vxx(self, t, x):
        return self.phi(t) * (self.p - 1) * x ** (self.p - 2)

    def pi_amount(self, x):
        """Montant (en unites monetaires) investi dans l'actif risque, pour une richesse x."""
        return self.pi_star * x


if __name__ == "__main__":
    m = MertonModel(r=0.02, mu=0.08, sigma=0.20, gamma=3.0, T=1.0)
    print(f"gamma={m.gamma}  p={m.p}  rho={m.rho:.6f}")
    print(f"pi* (fraction de richesse) = {m.pi_star:.6f}")
    print(f"V(0,1) = {m.V(0.0, 1.0):.6f}")
