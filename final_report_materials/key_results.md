# Key Results

Les tableaux sources propres sont dans `tables/`. Les valeurs ci-dessous proviennent du dernier rejeu des scripts finaux.

## Experience 1 - Merton HJB 1D

Profil final `Nx=401`, `Nt=1600` :

| gamma | erreur relative V(0,x0) | pi_num | pi_exact | RMSE controle interieur | max erreur controle | fraction Vxx>=0 | fraction clipping |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5 | 4.5313e-07 | 0.999997 | 1.000000 | 1.1241e-05 | 3.5872e-05 | 0.0 | 0.0 |
| 3.0 | 1.3532e-05 | 0.499950 | 0.500000 | 1.7893e-04 | 5.5113e-04 | 0.0 | 0.0 |
| 5.0 | 8.2464e-05 | 0.299856 | 0.300000 | 5.1518e-04 | 1.5869e-03 | 0.0 | 0.0 |

Conclusion : la HJB numerique retrouve tres precisement la solution analytique. La concavite attendue est respectee sur les profils finaux.

## Experience 2 - Merton empirique AAPL

Periode backtest : 2018-02-01 a 2025-12-31. Nombre de decisions mensuelles : 95. Premiere decision apres 504 observations.

Meilleure richesse finale : `benchmark_aapl_100`, richesse `6.933305`.

| strategie | richesse finale | rendement annualise | vol annualisee | Sharpe | max drawdown | exposition moyenne |
|---|---:|---:|---:|---:|---:|---:|
| merton_unconstrained_gamma_1.5 | 0.102646 | -0.250445 | 1.824560 | 0.243099 | -1.096752 | 2.298575 |
| merton_long_only_gamma_1.5 | 5.420355 | 0.238660 | 0.296542 | 0.780242 | -0.385159 | 0.900391 |
| merton_unconstrained_gamma_3 | 2.594233 | 0.128307 | 0.427798 | 0.434916 | -0.708414 | 1.113623 |
| merton_long_only_gamma_3 | 4.175722 | 0.198409 | 0.275923 | 0.697538 | -0.385155 | 0.784687 |
| merton_unconstrained_gamma_5 | 2.289491 | 0.110593 | 0.246734 | 0.440929 | -0.484726 | 0.664314 |
| merton_long_only_gamma_5 | 2.453257 | 0.120352 | 0.208036 | 0.522503 | -0.365095 | 0.586526 |
| benchmark_aapl_100 | 6.933305 | 0.277883 | 0.309081 | 0.861773 | -0.385159 | 1.000000 |
| benchmark_50_50_monthly | 3.175404 | 0.157562 | 0.152959 | 0.859113 | -0.206919 | 0.502375 |
| benchmark_cash_100 | 1.234085 | 0.026993 | 0.001264 | approx 0 | 0.000000 | 0.000000 |

Poids non contraints hors bornes :

| gamma | fraction w<0 | fraction w>1 |
|---:|---:|---:|
| 1.5 | 0.021053 | 0.789474 |
| 3.0 | 0.021053 | 0.484211 |
| 5.0 | 0.021053 | 0.147368 |

Certainty equivalents annuels gagnants :

| gamma | meilleure strategie CE | CE annualise |
|---:|---|---:|
| 1.5 | benchmark_aapl_100 | 0.244677 |
| 3.0 | benchmark_aapl_100 | 0.171902 |
| 5.0 | benchmark_50_50_monthly | 0.108403 |

Attention : la strategie non contrainte gamma 1.5 atteint une richesse terminale tres faible et un drawdown inferieur a -100%, ce qui rend l'interpretation CRRA fragile lorsque la richesse devient non positive.

## Experience 3A - Call europeen, benchmark heuristique

Sans couts, le delta hedging s'ameliore avec la frequence :

| frequence | RMSE delta lambda=0 |
|---:|---:|
| 12 | 1.945227 |
| 26 | 1.334097 |
| 52 | 0.952700 |
| 126 | 0.616434 |
| 252 | 0.435521 |

Meilleurs compromis heuristiques par lambda :

| lambda | best delta freq | best delta RMSE | best delta cout | best no-trade freq | band | RMSE no-trade | cout no-trade | score ameliore |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0.0000 | 252 | 0.435521 | 0.000000 | 252 | 0.00 | 0.435521 | 0.000000 | False |
| 0.0005 | 252 | 0.528463 | 0.274517 | 252 | 0.02 | 0.541496 | 0.230418 | True |
| 0.0020 | 52 | 1.136580 | 0.558176 | 126 | 0.05 | 1.033926 | 0.571897 | True |
| 0.0100 | 12 | 2.651388 | 1.600537 | 26 | 0.10 | 2.488746 | 1.630252 | True |

## Experience 3B - Call europeen, Bellman

Validation `lambda=0` :

| metrique | valeur |
|---|---:|
| N | 48 |
| Ns | 401 |
| quadrature | 15 |
| RMSE politique interieur vs delta BS | 0.003105831 |
| MAE politique interieur | 0.002014484 |
| max abs interieur | 0.027150528 |
| RMSE hedge Bellman | 0.965198958 |
| RMSE hedge delta meme frequence | 0.987382532 |
| ratio RMSE Bellman / delta | 0.977532949 |

Largeur HOLD moyenne avec couts :

| lambda | largeur HOLD |
|---:|---:|
| 0.0005 | 0.002184 |
| 0.0020 | 0.008569 |
| 0.0100 | 0.036458 |

Backtest couts selectionne :

| lambda | strategie | RMSE | cout moyen | turnover | trades | HOLD realise |
|---:|---|---:|---:|---:|---:|---:|
| 0.0005 | Bellman | 3.341914 | 0.077833 | 1.546063 | 4.682333 | 0.672727 |
| 0.0005 | delta BS meme freq | 1.937738 | 0.079903 | 1.582395 | 11.996333 | n/a |
| 0.0005 | best no-trade Exp3 | 0.538510 | 0.227468 | 4.532196 | 97.323667 | n/a |
| 0.0020 | Bellman | 3.331157 | 0.306670 | 1.522458 | 4.599667 | 0.654545 |
| 0.0020 | delta BS meme freq | 1.987228 | 0.319612 | 1.582395 | 11.996333 | n/a |
| 0.0020 | best no-trade Exp3 | 1.032470 | 0.564504 | 2.813107 | 28.928000 | n/a |
| 0.0100 | Bellman | 3.649685 | 1.312189 | 1.299667 | 3.503333 | 0.800000 |
| 0.0100 | delta BS meme freq | 2.659977 | 1.598061 | 1.582395 | 11.996333 | n/a |
| 0.0100 | best no-trade Exp3 | 2.493748 | 1.620721 | 1.612085 | 7.450667 | n/a |

Conclusion : sans couts, Bellman est valide. Avec couts, HOLD emerge et s'elargit, mais les RMSE ne sont pas competitifs; la branche est qualitativement utile mais numeriquement limitee.

## Experience 4 - Down-and-Out Call

Prix initial benchmark risque-neutre numerique : `8.838907479`.

Detection de barriere :

| methode | fraction knock-out |
|---|---:|
| naive endpoints | 0.1730 |
| Brownian bridge | 0.1855 |
| bridge - naive | 0.0125 |

Sans couts, frequence 24 :

| strategie | RMSE | MAE | biais | KO RMSE | survivant RMSE | near-barrier RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Bellman | 1.351964 | 0.982899 | 0.023536 | 1.144044 | 1.394991 | 1.390565 |
| delta barriere | 1.395295 | 1.046983 | -0.002311 | 0.968480 | 1.475337 | 1.361827 |
| delta europeen | 1.396655 | 1.044097 | -0.035478 | 0.914728 | 1.484701 | 1.379266 |

Convergence temporelle Bellman sans couts :

| frequence | RMSE Bellman | RMSE delta barriere |
|---:|---:|---:|
| 6 | 2.479035 | 2.675715 |
| 12 | 1.791438 | 1.926218 |
| 24 | 1.305568 | 1.374097 |

Profils pres de la barriere a t=0 :

| S | prix barriere | delta barriere | delta europeen | position Bellman | proba KO prochain pas |
|---:|---:|---:|---:|---:|---:|
| 80.8 | 0.277390 | 0.341501 | 0.193254 | 0.130404 | 0.887976 |
| 84.0 | 1.428266 | 0.320357 | 0.250866 | 0.342080 | 0.057465 |
| 100.0 | 8.832483 | 0.594169 | 0.579260 | 0.598420 | 0.000000 |
| 120.0 | 23.744620 | 0.863931 | 0.866847 | 0.870757 | 0.000000 |
| 150.0 | 52.061420 | 0.979601 | 0.987037 | 0.979810 | 0.000000 |

Branche couts pilote :

| lambda | Bellman RMSE | Bellman cout | Bellman turnover | hold fraction |
|---:|---:|---:|---:|---:|
| 0.0005 | 5.972320 | 0.035597 | 0.694100 | 0.156863 |
| 0.0020 | 5.900394 | 0.149286 | 0.732530 | 0.155338 |
| 0.0100 | 8.002441 | 0.697115 | 0.653523 | 0.143791 |

Conclusion : Bellman adapte clairement le hedge au risque de knock-out. La branche couts est seulement pilote et ne doit pas etre decrite comme convergee.
