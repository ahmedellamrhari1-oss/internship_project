# Equations

## Merton

Dynamique de richesse :

```text
dX_t = [r X_t + pi_t (mu-r)] dt + pi_t sigma dW_t
```

Utilite CRRA :

```text
U(x) = x^(1-gamma) / (1-gamma)
```

HJB :

```text
V_t + sup_pi [(r x + pi(mu-r)) V_x + 0.5 pi^2 sigma^2 V_xx] = 0
V(T,x) = U(x)
```

Controle local :

```text
pi* = - (mu-r)/sigma^2 * V_x / V_xx
```

PDE reduite :

```text
V_t + r x V_x - ((mu-r)^2 / (2 sigma^2)) * V_x^2 / V_xx = 0
```

Solution analytique :

```text
V_exact(t,x) =
x^(1-gamma)/(1-gamma)
* exp((1-gamma) * (r + (mu-r)^2/(2 gamma sigma^2)) * (T-t))
```

Controle analytique :

```text
pi_exact(t,x) = (mu-r)/(gamma sigma^2) * x
```

## Allocation empirique Merton

Estimation annualisee :

```text
sigma_hat_t = std(R) * sqrt(252)
mu_hat_t = 252 mean(R) + 0.5 sigma_hat_t^2
w*_t = (mu_hat_t - r_t) / (gamma sigma_hat_t^2)
```

Version long-only :

```text
w_t = min(1, max(0, w*_t))
```

## Hedging call europeen

Dynamique GBM exacte :

```text
S_{n+1} = S_n exp((mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z_{n+1})
```

Payoff call :

```text
H = (S_T-K)^+
```

Cout proportionnel :

```text
C_n = lambda S_n |a_n - q_n|
```

Dynamique cash :

```text
b_n^+ = b_n - (a_n-q_n) S_n - C_n
b_{n+1} = b_n^+ exp(r dt)
q_{n+1} = a_n
W_T = b_T + q_T S_T
```

Objectif Bellman :

```text
V_n(z) = min_a E[V_{n+1}(Z_{n+1}) | Z_n=z, a_n=a]
V_N = (W_T-H)^2
```

Reduction quadratique sans couts :

```text
V_n(S,W) = A_n(S) W^2 - 2 B_n(S) W + C_n(S)
```

## Down-and-Out Call

Indicateur de survie :

```text
I_t = 1 si la barriere n'a pas ete touchee
I_t = 0 sinon
```

Payoff :

```text
H = (S_T-K)^+ 1_{min_{0<=u<=T} S_u > B}
  = (S_T-K)^+ I_T
```

Probabilite de survie Brownian bridge conditionnelle aux endpoints :

```text
p_surv = 1 - exp(-2 log(S_n/B) log(S_{n+1}/B) / (sigma^2 dt))
```

si `S_n>B` et `S_{n+1}>B`; sinon `p_surv=0`.

Reduction quadratique sans couts :

```text
V_n(S,W,I) = A_n(S,I) W^2 - 2 B_n(S,I) W + C_n(S,I)
```

Objectif terminal :

```text
L_T = (W_T - (S_T-K)^+ I_T)^2
```
