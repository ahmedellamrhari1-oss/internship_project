# Final Report Materials

Ce dossier rassemble les elements necessaires pour rediger le rapport scientifique final du projet.

Ordre de lecture recommande :

1. `methodology.md` pour les questions scientifiques, modeles, et protocoles.
2. `equations.md` pour les equations effectivement utilisees dans le code.
3. `key_results.md` pour les valeurs numeriques finales a reporter.
4. `limitations.md` pour separer clairement ce qui est valide, exploratoire, ou non converge.
5. `tables/` pour les CSV propres utilises dans les tableaux du rapport.
6. `figures/` pour les figures retenues et renommees.
7. `configs/` pour les configs et summaries originaux recopies depuis `results/`.
8. `data_sources.md` pour les sources de donnees et la reproductibilite.

Les resultats originaux restent dans `results/`. Les figures originales ne sont pas supprimees.

Statut scientifique synthetique :

| Experience | Statut |
|---|---|
| Exp.1 Merton HJB 1D | VALIDE |
| Exp.2 Merton empirique AAPL | VALIDE AVEC LIMITES |
| Exp.3A Call europeen, benchmark heuristique couts | VALIDE |
| Exp.3B Call europeen, Bellman | VALIDE AVEC LIMITES |
| Exp.4 Down-and-Out Call, Bellman | EXPLORATOIRE |

Les branches pilotes avec couts multidimensionnels ne doivent pas etre presentees comme pleinement convergees.
