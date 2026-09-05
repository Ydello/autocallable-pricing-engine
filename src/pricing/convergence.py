"""
convergence.py
===============
Analyse de la convergence du prix Monte Carlo en fonction du nombre de trajectoires.
Utilisé dans notebooks/03_validation_pricing.ipynb pour justifier le choix de n_paths.

TODO (à finaliser ensemble) :
- Définir le critère de convergence acceptable (ex. std_error < X% du prix)
"""

from dataclasses import dataclass

import numpy as np

from pricing.monte_carlo_engine import MonteCarloEngine


@dataclass
class ConvergenceStudyResult:
    path_counts: np.ndarray
    prices: np.ndarray
    std_errors: np.ndarray


def run_convergence_study(
    engine: MonteCarloEngine,
    path_counts: list[int],
    random_seed: int = 42,
) -> ConvergenceStudyResult:
    """
    Reprice le produit pour différentes tailles d'échantillon afin d'observer
    la stabilisation du prix et la réduction de l'erreur standard.

    Parameters
    ----------
    engine : MonteCarloEngine
        Moteur de pricing déjà configuré (specs + market_params)
    path_counts : list[int]
        Liste croissante de tailles d'échantillon à tester, ex. [1000, 5000, 10000, 50000, 100000]

    Returns
    -------
    ConvergenceStudyResult
    """
    prices = []
    std_errors = []

    for n in path_counts:
        result = engine.price(n_paths=n, random_seed=random_seed, antithetic=(n % 2 == 0))
        prices.append(result.price)
        std_errors.append(result.std_error)

    return ConvergenceStudyResult(
        path_counts=np.array(path_counts),
        prices=np.array(prices),
        std_errors=np.array(std_errors),
    )


def required_paths_for_precision(
    engine: MonteCarloEngine,
    target_std_error: float,
    initial_n: int = 10_000,
    random_seed: int = 42,
) -> int:
    """
    Estime (par extrapolation en 1/sqrt(n)) le nombre de trajectoires nécessaires
    pour atteindre une erreur standard cible, à partir d'un run initial.

    L'erreur standard décroît en 1/sqrt(n), donc :
        n_requis = initial_n * (std_error_initial / target_std_error)^2

    Returns
    -------
    int
        Nombre de trajectoires estimé nécessaire
    """
    result = engine.price(n_paths=initial_n, random_seed=random_seed)
    ratio = (result.std_error / target_std_error) ** 2
    return int(np.ceil(initial_n * ratio))
