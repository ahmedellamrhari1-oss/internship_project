# Limitations

## Hypotheses theoriques

- Les modeles de Merton et Black-Scholes supposent volatilite constante, marche frictionless hors couts explicites, et dynamique lognormale.
- Les benchmarks delta utilisent des formules ou approximations risque-neutres, tandis que les backtests simulent sous drift reel `mu=0.08`.
- L'utilite CRRA est delicate si la richesse devient negative ou nulle; les certainty equivalents empiriques doivent etre interpretes avec prudence pour les strategies fortement leverees.

## Limitations empiriques

- L'Experience 2 depend fortement de l'estimation du drift, connue pour etre instable.
- Le levier de la strategie Merton non contrainte produit des expositions extremes, notamment pour gamma faible.
- L'etude AAPL est mono-actif et ne demontre pas une robustesse multi-actifs ou multi-periodes economiques.

## Limitations numeriques

- Experience 1 : le temps ne donne pas une monotonie parfaite de l'erreur, meme si les erreurs restent faibles; l'espace converge proprement.
- Experience 3B avec couts : la DP explicite `(S,b,q)` reste couteuse. La region HOLD est qualitative et interpretable, mais les RMSE Bellman avec couts ne battent pas les benchmarks.
- Experience 4 avec couts : la branche `(S,b,q,I)` est seulement pilote. Elle reduit le turnover mais ne fournit pas une politique convergee.
- Les grilles multidimensionnelles avec cash peuvent induire des erreurs d'interpolation et de controle; les diagnostics de frontiere sont reportes mais ne remplacent pas une etude de convergence exhaustive.

## Resultats non pleinement converges

- Exp.3B `lambda>0` : VALIDEE qualitativement pour l'emergence HOLD, non pleinement convergee quantitativement.
- Exp.4 `lambda>0` : EXPLORATOIRE; ne pas presenter comme validation d'une strategie optimale avec couts.
- Exp.4 delta barriere : benchmark numerique par grille risque-neutre, pas formule fermee analytique.
