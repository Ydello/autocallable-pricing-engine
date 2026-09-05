# Autocallable Structured Product — Pricing & Risk Analysis

Pricing d'un Phoenix Autocall mono-sous-jacent par simulation Monte Carlo,
avec analyse des Greeks et stress tests.

## Structure

```
config/     -> paramètres du produit (product_config.yaml)
data/       -> données brutes et traitées
src/        -> code source (voir détail ci-dessous)
notebooks/  -> notebooks d'exploration, calibration, validation, résultats
tests/      -> tests unitaires
outputs/    -> figures et rapports générés
```

## Modules (src/)

| Module | Rôle | Statut |
|---|---|---|
| `data_loader.py` | Récupération données marché (yfinance, FRED) | TODO données réelles |
| `market_calibration.py` | Calibration vol / taux / dividende | TODO |
| `simulation/gbm_simulator.py` | Génération trajectoires GBM | Squelette fonctionnel |
| `simulation/variance_reduction.py` | Réduction de variance | Squelette fonctionnel |
| `product/product_specs.py` | Spécifications du produit | Fonctionnel (dépend config) |
| `product/autocall_payoff.py` | Logique payoff Phoenix | Fonctionnel |
| `pricing/monte_carlo_engine.py` | Moteur de pricing | Fonctionnel |
| `pricing/convergence.py` | Analyse de convergence | Fonctionnel |
| `risk/greeks.py` | Calcul des Greeks | Fonctionnel |
| `risk/stress_testing.py` | Scénarios de stress | Fonctionnel |
| `utils/discounting.py` | Actualisation | Fonctionnel |
| `utils/validation.py` | Tests de cohérence | Squelette |

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation rapide (dans un notebook)

```python
import sys
sys.path.append("../src")   # selon l'emplacement du notebook

from product.product_specs import ProductSpecs
from pricing.monte_carlo_engine import MonteCarloEngine

specs = ProductSpecs.from_config("../config/product_config.yaml")
engine = MonteCarloEngine(specs)
result = engine.price()
print(result)
```

## Prochaines étapes (données manquantes à finaliser)

- [ ] Choix définitif du sous-jacent (ticker exact)
- [ ] Calibration du taux sans risque (FRED)
- [ ] Calibration de la volatilité (VIX/VSTOXX ou surface d'options)
- [ ] Valeurs exactes des barrières (step-down, coupon, knock-in)
- [ ] Date de valorisation (pricing_date)
