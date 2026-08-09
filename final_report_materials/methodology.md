# Methodology

## Experience 1 - Merton / HJB 1D

Question scientifique : verifier qu'une resolution HJB par differences finies retrouve la solution analytique du probleme de Merton 1D.

Modele : richesse auto-financee avec actif risque et actif sans risque. L'utilite terminale est CRRA. Le controle est le montant investi dans l'actif risque.

Etat : richesse positive. Le solveur numerique travaille en log-richesse pour stabiliser la grille.

Objectif : maximiser l'esperance d'utilite terminale.

Methode numerique : resolution backward de la PDE HJB reduite, calcul de `V_x` et `V_xx`, reconstruction de `pi_HJB`, diagnostics de concavite et de clipping.

Validation : comparaison a la valeur analytique et au controle analytique pour gamma 1.5, 3 et 5, avec sweeps en espace et en temps.

## Experience 2 - Merton empirique AAPL

Question scientifique : tester hors echantillon si l'allocation de Merton reste performante sur donnees reelles lorsque les parametres sont estimes.

Donnees : AAPL ajuste, 2016-01-01 a 2025-12-31, et taux FRED DGS3MO forward-fill. Les copies brutes sont conservees dans `results/merton_aapl_empirical/data/raw/`.

Etat : donnees historiques disponibles a chaque date de decision.

Controle : poids AAPL calcule par la formule de Merton, en version non contrainte et long-only.

Objectif : evaluation empirique de richesse, risque, Sharpe, drawdown, turnover, exposition, utilite CRRA et certainty equivalent.

Methode : estimation walk-forward sur fenetre glissante de 504 seances, reequilibrage mensuel, sans look-ahead.

Validation : tests de non-utilisation d'information future, bornes long-only, benchmark cash, absence de NaN/Inf, reproductibilite.

## Experience 3A - Hedging call europeen, benchmark heuristique

Question scientifique : mesurer le compromis entre precision de replication et couts de transaction pour un call europeen.

Modele : Black-Scholes avec simulation GBM exacte sous drift reel pour les backtests.

Controle : position delta discrete reequilibree a plusieurs frequences, avec ou sans no-trade band heuristique.

Objectif : erreur terminale de hedge et couts realises. Le critere mixte `J` est utilise dans cette sous-experience heuristique pour choisir une bande, mais il est interprete comme score pratique, pas comme objectif de controle fondamental.

Validation : sans couts, l'erreur de delta hedging diminue quand la frequence augmente. Avec couts, les bandes no-trade peuvent reduire le score cout/erreur.

## Experience 3B - Hedging call europeen, Bellman

Question scientifique : verifier que Bellman retrouve le delta Black-Scholes sans couts, puis etudier l'emergence d'une zone HOLD avec couts.

Etat sans couts : `(S,W)`. Une reduction quadratique exacte elimine la dimension cash/wealth.

Etat avec couts : `(S,b,q)`, grille explicite. Le controle est la nouvelle position `a=q'`.

Objectif : minimiser `E[(W_T-H)^2]`; avec couts, ceux-ci sont deduits directement du cash.

Methode : programmation dynamique discrete backward. La politique Bellman n'utilise pas le delta BS pendant l'optimisation.

Validation : le cas `lambda=0` est valide contre le delta BS. La branche `lambda>0` montre une region HOLD croissante mais reste quantitativement moins convergee.

## Experience 4 - Down-and-Out Call

Question scientifique : tester si Bellman reste utilisable pour une option path-dependent avec risque de knock-out.

Etat sans couts : `(S,W,I)`, avec `I=1` si l'option est vivante et `I=0` apres knock-out. Reduction quadratique exacte en richesse.

Etat avec couts : `(S,b,q,I)`, grille explicite pilote.

Controle : nouvelle position en action `a=q'`.

Objectif : minimiser `E[(W_T-(S_T-K)^+ I_T)^2]`.

Traitement de la barriere : probabilite de survie Brownian bridge entre deux dates, utilisee dans Bellman et dans les backtests.

Benchmarks : delta europeen, delta barriere numerique risque-neutre, et no-trade applique au delta barriere.

Validation : sans couts, Bellman est coherent et proche des benchmarks. Avec couts, la branche est pilote et non pleinement convergee.
