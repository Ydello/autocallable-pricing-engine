"""
stress_testing.py
===================
Applique des scénarios de stress au moteur de pricing pour évaluer la robustesse
du produit face à des chocs de marché combinés (vol, spot, taux).

TODO (à finaliser ensemble) :
- Définir les scénarios historiques précis à répliquer (ex. 2008 : quels chocs
  exacts de vol et de spot appliquer ? mars 2020 : idem)
- Décider si on stress aussi la corrélation (non applicable en mono-sous-jacent,
  mais à garder en tête si extension vers un panier)
"""

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from product.product_specs import ProductSpecs
from simulation.gbm_simulator import GBMParameters
from pricing.monte_carlo_engine import MonteCarloEngine


@dataclass
class StressScenario:
    name: str
    spot_shock_pct: float = 0.0        # ex. -0.30 pour un crash de -30%
    vol_shock_abs: float = 0.0         # ex. +0.20 pour +20 points de vol
    rate_shock_abs: float = 0.0        # ex. -0.01 pour -100bp


@dataclass
class StressTestResult:
    scenario_name: str
    price: float
    price_change_pct: float
    knock_in_probability: float
    knock_in_probability_change: float


# TODO : affiner ces scénarios avec des chocs historiquement documentés
DEFAULT_SCENARIOS = [
    StressScenario(name="Choc de volatilité modéré (+10 vol pts)", vol_shock_abs=0.10),
    StressScenario(name="Choc de volatilité sévère (+25 vol pts)", vol_shock_abs=0.25),
    StressScenario(name="Crash type 2008", spot_shock_pct=-0.40, vol_shock_abs=0.30),
    StressScenario(name="Crash type mars 2020", spot_shock_pct=-0.30, vol_shock_abs=0.35),
    StressScenario(name="Hausse des taux (+100bp)", rate_shock_abs=0.01),
    StressScenario(name="Baisse des taux (-100bp)", rate_shock_abs=-0.01),
]


def apply_scenario(market_params: GBMParameters, scenario: StressScenario) -> GBMParameters:
    """
    Applique les chocs d'un scénario aux paramètres de marché de référence.

    Important : `reference_spot` (niveau S0 des barrières, fixé à l'émission) est
    explicitement figé à sa valeur pré-choc — seul le spot COURANT bouge. C'est ce qui
    modélise correctement un crash de marché survenant APRÈS l'émission du produit :
    les barrières contractuelles ne bougent pas, seul le marché chute autour d'elles.
    """
    frozen_reference = market_params.effective_reference_spot()
    return replace(
        market_params,
        spot=market_params.spot * (1 + scenario.spot_shock_pct),
        reference_spot=frozen_reference,
        volatility=max(market_params.volatility + scenario.vol_shock_abs, 0.001),
        risk_free_rate=market_params.risk_free_rate + scenario.rate_shock_abs,
    )


def run_stress_tests(
    specs: ProductSpecs,
    base_market_params: GBMParameters,
    scenarios: Optional[list[StressScenario]] = None,
    n_paths: int = 100_000,
    random_seed: int = 42,
) -> list[StressTestResult]:
    """
    Reprice le produit sous chaque scénario de stress et compare au prix de référence.

    Parameters
    ----------
    specs : ProductSpecs
    base_market_params : GBMParameters
        Paramètres de marché de référence (non stressés)
    scenarios : list[StressScenario], optional
        Liste de scénarios à tester. Si None, utilise DEFAULT_SCENARIOS.
    n_paths : int
    random_seed : int
        Fixé pour permettre une comparaison cohérente entre scénarios (common random numbers)

    Returns
    -------
    list[StressTestResult]
    """
    scenarios = scenarios if scenarios is not None else DEFAULT_SCENARIOS

    base_engine = MonteCarloEngine(specs, base_market_params)
    base_result = base_engine.price(n_paths=n_paths, random_seed=random_seed)

    results = []
    for scenario in scenarios:
        stressed_params = apply_scenario(base_market_params, scenario)
        stressed_engine = MonteCarloEngine(specs, stressed_params)
        stressed_result = stressed_engine.price(n_paths=n_paths, random_seed=random_seed)

        price_change_pct = (stressed_result.price - base_result.price) / base_result.price * 100
        ki_change = stressed_result.knock_in_probability - base_result.knock_in_probability

        results.append(
            StressTestResult(
                scenario_name=scenario.name,
                price=stressed_result.price,
                price_change_pct=price_change_pct,
                knock_in_probability=stressed_result.knock_in_probability,
                knock_in_probability_change=ki_change,
            )
        )

    return results
