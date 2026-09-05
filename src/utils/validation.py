"""
validation.py
==============
Tests de cohérence économique et de robustesse numérique du modèle.
Utilisé dans notebooks/03_validation_pricing.ipynb.

TODO (à finaliser ensemble) :
- Ajouter des cas de test avec solution semi-analytique connue si possible
  (ex. produit dégénéré équivalent à une obligation zéro-coupon quand les
  barrières sont mises à des valeurs extrêmes)
"""

from typing import Callable

import numpy as np


def check_monotonicity_vs_volatility(
    pricing_fn: Callable[[float], float],
    vol_grid: np.ndarray,
) -> dict:
    """
    Vérifie comment le prix évolue avec la volatilité.

    Pour un autocall, l'effet n'est PAS monotone de façon triviale : une vol plus élevée
    augmente à la fois le risque de knock-in (baisse le prix) et la probabilité de
    toucher des extrêmes hauts (peut affecter l'autocall). Ce test sert surtout à
    repérer un comportement aberrant (ex. discontinuité brutale, valeur négative).

    Parameters
    ----------
    pricing_fn : Callable
        Fonction qui prend une volatilité et retourne un prix
    vol_grid : np.ndarray
        Grille de volatilités à tester, ex. np.linspace(0.05, 0.50, 10)

    Returns
    -------
    dict
        {"vols": ..., "prices": ..., "is_smooth": bool}

    TODO : définir un seuil précis de "saut anormal" une fois le moteur de pricing
           branché sur des données réelles
    """
    prices = np.array([pricing_fn(v) for v in vol_grid])
    diffs = np.diff(prices)
    max_jump = np.max(np.abs(diffs)) if len(diffs) > 0 else 0.0
    is_smooth = max_jump < 0.1 * np.mean(prices)  # seuil arbitraire, TODO à calibrer
    return {"vols": vol_grid, "prices": prices, "is_smooth": is_smooth}


def check_price_bounds(price: float, nominal: float) -> None:
    """
    Vérifie que le prix reste dans des bornes économiquement plausibles :
    - Ne doit pas être négatif
    - Ne doit pas dépasser trivialement le nominal + somme de tous les coupons max possibles

    TODO : affiner la borne haute exacte une fois le coupon maximum théorique connu
    """
    if price < 0:
        raise ValueError(f"Prix négatif détecté : {price}. Vérifier le moteur de pricing.")
    if price > nominal * 2:  # borne large arbitraire, TODO à resserrer
        raise ValueError(f"Prix ({price}) anormalement élevé par rapport au nominal ({nominal}).")


def check_monte_carlo_convergence(
    running_means: np.ndarray,
    tolerance_pct: float = 0.5,
) -> bool:
    """
    Vérifie que la moyenne Monte Carlo s'est stabilisée sur la fin de la simulation
    (les derniers X% de trajectoires ne changent plus significativement la moyenne).

    Parameters
    ----------
    running_means : np.ndarray
        Moyenne cumulative du payoff après k trajectoires, pour k croissant
    tolerance_pct : float
        Écart relatif toléré (en %) entre la moyenne finale et la moyenne à mi-parcours

    Returns
    -------
    bool
        True si convergence jugée satisfaisante
    """
    final_mean = running_means[-1]
    mid_mean = running_means[len(running_means) // 2]
    relative_diff = abs(final_mean - mid_mean) / abs(final_mean) * 100
    return relative_diff < tolerance_pct
