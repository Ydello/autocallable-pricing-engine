"""
variance_reduction.py
======================
Techniques de réduction de variance pour améliorer la précision du Monte Carlo
à nombre de trajectoires fixé.

Techniques couvertes :
- Variables antithétiques (déjà intégrée dans gbm_simulator.simulate_gbm_paths)
- Variable de contrôle (control variate)
- Placeholder pour quasi-Monte Carlo (Sobol)

TODO (à finaliser ensemble) :
- Choisir une variable de contrôle pertinente (ex. le prix d'une obligation zéro-coupon
  simple, ou le prix d'un call européen sous Black-Scholes fermé, si applicable)
- Décider si on implémente Sobol (utile surtout en faible dimension, à évaluer
  selon le nombre de dates d'observation)
"""

from typing import Callable, Optional

import numpy as np


def control_variate_adjustment(
    payoffs: np.ndarray,
    control_values: np.ndarray,
    control_true_mean: float,
) -> tuple[np.ndarray, float]:
    """
    Applique une correction par variable de contrôle.

    Principe : si X est le payoff qui nous intéresse et Y une variable corrélée
    dont on connaît l'espérance théorique exacte E[Y], on calcule :

        X_corrigé = X - beta * (Y - E[Y])

    avec beta = Cov(X, Y) / Var(Y), ce qui minimise la variance du résultat corrigé.

    Parameters
    ----------
    payoffs : np.ndarray
        Payoffs simulés du produit (une valeur par trajectoire)
    control_values : np.ndarray
        Valeurs simulées de la variable de contrôle (même trajectoires)
    control_true_mean : float
        Espérance théorique exacte de la variable de contrôle

    Returns
    -------
    tuple
        (payoffs_corrigés, beta_utilisé)

    TODO : brancher une variable de contrôle concrète une fois choisie
    """
    cov_matrix = np.cov(payoffs, control_values)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    adjusted = payoffs - beta * (control_values - control_true_mean)
    return adjusted, float(beta)


def sobol_normal_samples(n_paths: int, n_dims: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Génère des échantillons normaux via séquence de Sobol (quasi-Monte Carlo),
    utile pour accélérer la convergence en dimension modérée.

    Returns
    -------
    np.ndarray
        Tableau (n_paths, n_dims) d'échantillons N(0,1) quasi-aléatoires

    TODO : implémenter avec scipy.stats.qmc.Sobol si cette technique est retenue
           (à activer seulement si la réduction de variance simple ne suffit pas)
    """
    raise NotImplementedError(
        "TODO: implémenter via scipy.stats.qmc.Sobol si nécessaire. "
        "Non prioritaire — activer seulement après validation du MC standard."
    )


def estimate_variance_reduction_gain(
    payoffs_standard: np.ndarray,
    payoffs_reduced: np.ndarray,
) -> float:
    """
    Compare la variance avant/après réduction de variance, pour quantifier le gain.

    Returns
    -------
    float
        Ratio de réduction de variance (ex. 3.0 = variance divisée par 3)
    """
    var_standard = np.var(payoffs_standard, ddof=1)
    var_reduced = np.var(payoffs_reduced, ddof=1)
    if var_reduced == 0:
        return float("inf")
    return float(var_standard / var_reduced)
