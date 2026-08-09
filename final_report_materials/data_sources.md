# Data Sources

## Experience 1

Pas de donnees externes. Les resultats sont generes par `exp_hjb_merton_1d.py` avec les parametres sauvegardes dans `results/hjb_merton_1d/config.json`.

## Experience 2

Donnees utilisees :

- AAPL prix quotidiens ajustes, copie brute : `results/merton_aapl_empirical/data/raw/aapl_yahoo_chart_raw.json`.
- Taux sans risque DGS3MO, copie brute Federal Reserve H15 : `results/merton_aapl_empirical/data/raw/federal_reserve_h15_treasury_constant_maturities_raw.csv`.
- Donnees journalieres nettoyees : `results/merton_aapl_empirical/data/backtest_daily_data.csv`.
- Serie risk-free forward-fill : `results/merton_aapl_empirical/data/risk_free_dgs3mo_daily.csv`.

Le dernier rejeu a reutilise les fichiers locaux, sans telechargement reseau.

## Experiences 3 et 4

Pas de donnees externes. Les trajectoires sont simulees avec GBM exact et seeds sauvegardees dans les configs :

- `results/hedging_transaction_costs/config.json`
- `results/hedging_bellman/config.json`
- `results/barrier_hedging_bellman/config.json`

## Reproductibilite

Scripts finaux :

- `exp_hjb_merton_1d.py`
- `exp_merton_aapl_empirical.py`
- `exp_hedging_transaction_costs.py`
- `exp_hedging_bellman.py`
- `exp_barrier_hedging_bellman.py`

Tests :

```text
python tests/test_core.py
```
